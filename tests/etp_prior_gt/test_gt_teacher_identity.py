"""The GT-teacher loader refuses a checkpoint trained as a different policy or
on a different map cache.  Teacher and student share the architecture, so a
student checkpoint would otherwise load silently and the run would distil the
student into itself.

The trainer module imports habitat, so the function is lifted from the source
with ast instead of being imported.
"""

import ast
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "vlnce_baselines" / "ss_trainer_ETP_PriorGT.py").read_text()
_FN = next(
    n for n in ast.parse(SRC).body
    if isinstance(n, ast.FunctionDef) and n.name == "_check_teacher_identity"
)
_NS = {}
exec(compile(ast.Module([_FN], type_ignores=[]), "ss_trainer_ETP_PriorGT.py", "exec"), _NS)  # noqa: S102
check = _NS["_check_teacher_identity"]

GT = ("PriorGTTry5Policy", "gt.legacy.r1p5.direction5.v1")


def _cfg(policy, namespace):
    return types.SimpleNamespace(
        IL=types.SimpleNamespace(gt_teacher_policy_name=policy, gt_teacher_map_namespace=namespace)
    )


def _ckpt(policy, namespace):
    return {
        "config": types.SimpleNamespace(
            MODEL=types.SimpleNamespace(
                policy_name=policy, MAP_ENCODER=types.SimpleNamespace(cache_namespace=namespace)
            )
        ),
        "state_dict": {},
    }


def test_matching_teacher_passes():
    check(_cfg(*GT), _ckpt(*GT), "t.pth")


def test_student_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="LLMGridTry5Policy"):
        check(_cfg(*GT), _ckpt("LLMGridTry5Policy", "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"), "t.pth")


def test_other_gt_namespace_is_rejected():
    with pytest.raises(ValueError, match="cache_namespace"):
        check(_cfg(*GT), _ckpt("PriorGTTry5Policy", "gt.online121c369.r1p5.direction5.v1"), "t.pth")


def test_checkpoint_without_config_passes():
    check(_cfg(*GT), {"state_dict": {}}, "t.pth")
