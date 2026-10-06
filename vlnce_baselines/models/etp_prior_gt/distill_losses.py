"""Loss terms for distilling a GT-map teacher into an LLM-map student.

The plain action-level KL (``action_kl``) asks the student to match the
teacher's whole action distribution.  Most of what makes the teacher better
is information the student cannot see (the GT route corridor), so that target
is largely noise to it.  The three terms below separate the part that can be
transferred:

* ``effect_match`` compares *how the map changes the decision* in teacher and
  student.  Both are evaluated with their own map and with a counterfactual
  map (another episode's raster, own metadata).  Vision and language are the
  same in both branches and cancel in the difference, so the only way for the
  student to reduce this loss is through its map pathway.
* ``js_divergence`` / ``gate_weights`` measure, per step, how much the LLM map
  misleads the *teacher* (teacher on GT map vs teacher on the LLM map).  Where
  the teacher itself goes wrong on the LLM map, the teacher's advantage is
  unreachable for the student and the step is down-weighted (never dropped:
  ``min_weight`` is the floor).
* ``factorized_kl`` splits the action KL into a STOP Bernoulli term and the
  conditional distribution over the non-STOP candidates, so the two can be
  weighted separately.

All functions take raw logits of shape (B, G) over the student's graph
(index 0 is STOP) and a boolean ``valid`` mask of the same shape.  Logits
outside ``valid`` are ignored: every softmax is renormalised over the valid
set only.  Rows without a valid entry contribute zero.  Everything is
computed in float32 regardless of the input dtype.
"""

from __future__ import annotations

from typing import Sequence

import torch

STOP_INDEX = 0
_NEG_INF = float("-inf")


def _check_pair(logits: torch.Tensor, valid: torch.Tensor, name: str) -> None:
    if logits.dim() != 2:
        raise ValueError(f"{name} must be (B, G), got shape {tuple(logits.shape)}")
    if valid.shape != logits.shape:
        raise ValueError(
            f"valid mask shape {tuple(valid.shape)} does not match {name} "
            f"{tuple(logits.shape)}"
        )
    if valid.dtype != torch.bool:
        raise TypeError(f"valid mask must be bool, got {valid.dtype}")


def masked_log_softmax(
    logits: torch.Tensor, valid: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Log-softmax over the valid entries only; invalid entries become -inf.

    Rows with no valid entry return all -inf (callers mask them out).
    """
    _check_pair(logits, valid, "logits")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = logits.float() / temperature
    z = torch.where(valid, z, torch.full_like(z, _NEG_INF))
    # logsumexp of an all -inf row is -inf; (-inf) - (-inf) = nan.  Guard it.
    lse = torch.logsumexp(z, dim=1, keepdim=True)
    has_valid = valid.any(dim=1, keepdim=True)
    lse = torch.where(has_valid, lse, torch.zeros_like(lse))
    return z - lse


def _masked_probs(logp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid, logp.exp(), torch.zeros_like(logp))


def action_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-row KL(teacher || student) over the valid set, scaled by T^2.

    Returns shape (B,).
    """
    t_logp = masked_log_softmax(teacher_logits, valid, temperature)
    s_logp = masked_log_softmax(student_logits, valid, temperature)
    t_p = _masked_probs(t_logp, valid)
    kl = torch.where(valid, t_p * (t_logp - s_logp), torch.zeros_like(t_p))
    return kl.sum(dim=1) * (temperature * temperature)


def factorized_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the action KL into a STOP term and a move-conditional term.

    Returns ``(stop_kl, move_kl, p_move)``, each of shape (B,):

    * ``stop_kl``: KL between the teacher's and the student's Bernoulli(STOP).
    * ``move_kl``: KL between the two distributions over the non-STOP valid
      candidates, each renormalised to exclude STOP.  Zero when STOP is the
      only valid action.
    * ``p_move``: the teacher's probability of not stopping.

    Identity: ``stop_kl + p_move * move_kl == action_kl`` (chain rule), so the
    caller re-weights the two parts, it never changes what is measured.
    """
    _check_pair(teacher_logits, valid, "teacher_logits")
    has_valid = valid.any(dim=1)
    if not valid[has_valid, STOP_INDEX].all():
        # The policy never masks STOP (index 0 is never "visited"); a masked
        # STOP would make the Bernoulli term meaningless.  Rows with no valid
        # entry at all (ignored decisions) are allowed and contribute zero.
        bad = (has_valid & ~valid[:, STOP_INDEX]).nonzero().flatten().tolist()
        raise ValueError(f"STOP is masked out in rows {bad}")
    t_logp = masked_log_softmax(teacher_logits, valid, temperature)
    s_logp = masked_log_softmax(student_logits, valid, temperature)
    t_stop = t_logp[:, STOP_INDEX].exp().clamp(1e-6, 1 - 1e-6)
    s_stop = s_logp[:, STOP_INDEX].exp().clamp(1e-6, 1 - 1e-6)
    stop_kl = t_stop * (t_stop.log() - s_stop.log()) + (1 - t_stop) * (
        (1 - t_stop).log() - (1 - s_stop).log()
    )
    stop_kl = torch.where(has_valid, stop_kl, torch.zeros_like(stop_kl))

    move_valid = valid.clone()
    move_valid[:, STOP_INDEX] = False
    t_move_logp = masked_log_softmax(teacher_logits, move_valid, temperature)
    s_move_logp = masked_log_softmax(student_logits, move_valid, temperature)
    t_move_p = _masked_probs(t_move_logp, move_valid)
    move_kl = torch.where(
        move_valid, t_move_p * (t_move_logp - s_move_logp), torch.zeros_like(t_move_p)
    ).sum(dim=1)
    scale = temperature * temperature
    return stop_kl * scale, move_kl * scale, 1 - t_stop


def _centered_delta(
    full_logits: torch.Tensor, cf_logits: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Map effect on the logits, mean-removed over the valid set.

    log-softmax differences and raw logit differences agree up to a per-row
    constant, which the centering removes, so raw logits are used.
    """
    # Masked logits are -inf on both sides; subtract after zeroing them so no
    # inf - inf appears even in the unselected branch of the where.
    zero = torch.zeros_like(full_logits, dtype=torch.float32)
    full = torch.where(valid, full_logits.float(), zero)
    cf = torch.where(valid, cf_logits.float(), zero)
    delta = full - cf
    count = valid.sum(dim=1, keepdim=True).clamp(min=1).float()
    mean = delta.sum(dim=1, keepdim=True) / count
    return torch.where(valid, delta - mean, torch.zeros_like(delta))


def effect_match(
    teacher_full: torch.Tensor,
    teacher_cf: torch.Tensor,
    student_full: torch.Tensor,
    student_cf: torch.Tensor,
    valid: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Per-row squared error between the student's and the teacher's map effect.

    ``reduction`` "sum" adds the squared errors over the valid candidates (so
    the term scales like the KL terms, per decision); "mean" divides by the
    number of valid candidates, which keeps it independent of the graph size.

    ``*_full`` are logits with each model's own map, ``*_cf`` with the
    counterfactual map.  The teacher side and ``student_cf`` are detached:
    gradient reaches the student only through ``student_full``, so the
    student can only lower this loss by changing what its map pathway adds.
    Returns shape (B,), the sum over valid entries (so it scales like the
    KL terms, per decision rather than per candidate).
    """
    _check_pair(teacher_full, valid, "teacher_full")
    _check_pair(student_full, valid, "student_full")
    delta_t = _centered_delta(teacher_full.detach(), teacher_cf.detach(), valid)
    delta_s = _centered_delta(student_full, student_cf.detach(), valid)
    sq = ((delta_s - delta_t) ** 2).sum(dim=1)
    if reduction == "sum":
        return sq
    if reduction == "mean":
        return sq / valid.sum(dim=1).clamp(min=1).float()
    raise ValueError(f"reduction must be sum or mean, got {reduction!r}")


def js_divergence(
    logits_a: torch.Tensor, logits_b: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Jensen-Shannon divergence (natural log, in [0, ln 2]) per row."""
    a_logp = masked_log_softmax(logits_a, valid)
    b_logp = masked_log_softmax(logits_b, valid)
    a_p = _masked_probs(a_logp, valid)
    b_p = _masked_probs(b_logp, valid)
    m_p = 0.5 * (a_p + b_p)
    m_logp = torch.where(valid, m_p.clamp(min=1e-12).log(), torch.zeros_like(m_p))

    def _kl(p, logp):
        return torch.where(valid, p * (logp - m_logp), torch.zeros_like(p)).sum(dim=1)

    return (0.5 * (_kl(a_p, a_logp) + _kl(b_p, b_logp))).clamp(min=0.0)


def gate_weights(
    divergence: torch.Tensor,
    tau: float,
    min_weight: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Soft reachability gate ``w = min_weight + (1 - min_weight) * exp(-d / tau)``.

    ``tau`` is an absolute scale and must be positive: a batch-relative scale
    (e.g. the batch median) would always push half of every batch below
    ``exp(-1)`` whatever the absolute reachability, and a batch here is only a
    handful of episodes.  Pick it from the JS values the plain-KL arm logs.

    With ``normalize`` the weights are rescaled to mean 1 over the batch, so
    the gate only redistributes the loss between steps and leaves its total
    (hence the effective distillation weight) equal to the ungated arm.
    ``divergence`` is detached: the gate is a weight, not a training target.
    """
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
    if tau <= 0:
        raise ValueError(f"tau must be positive (an absolute JS scale), got {tau}")
    d = divergence.detach().float()
    w = min_weight + (1.0 - min_weight) * torch.exp(-d / float(tau))
    if normalize and w.numel() > 0:
        w = w / w.mean()
    return w


def donor_permutation(scene_ids: Sequence[str]) -> list[int] | None:
    """A derangement of the batch used to pick each episode's donor raster.

    Among the cyclic shifts the one pairing the most episodes with a
    different scene is chosen (ties: the smallest shift), so the
    counterfactual map is in-distribution (a real LLM raster) but carries no
    information about the episode's own scene wherever the batch allows.
    Returns ``None`` for a batch of one, where no derangement exists.
    """
    n = len(scene_ids)
    if n < 2:
        return None
    best_shift, best_mismatch = 1, -1
    for shift in range(1, n):
        mismatch = sum(
            scene_ids[i] != scene_ids[(i + shift) % n] for i in range(n)
        )
        if mismatch > best_mismatch:
            best_shift, best_mismatch = shift, mismatch
    return [(i + best_shift) % n for i in range(n)]
