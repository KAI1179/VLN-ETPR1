"""G1 registration: masking, actual navigation invariance, and feedback use."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from transformers import BertConfig, PretrainedConfig

ROOT = Path(__file__).resolve().parents[2]


def load_modules(monkeypatch):
    for package in (
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
        "vlnce_baselines.common",
    ):
        module = types.ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    for name, relative in (
        ("vlnce_baselines.common.transformer", "vlnce_baselines/common/transformer.py"),
        ("vlnce_baselines.common.ops", "vlnce_baselines/common/ops.py"),
        (
            "vlnce_baselines.models.etp_prior_gt.map_fusion",
            "vlnce_baselines/models/etp_prior_gt/map_fusion.py",
        ),
        (
            "vlnce_baselines.models.etp_prior_gt.registration_head",
            "vlnce_baselines/models/etp_prior_gt/registration_head.py",
        ),
        (
            "vlnce_baselines.models.etp_prior_gt.vilmodel_cmt",
            "vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py",
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return (
        sys.modules["vlnce_baselines.models.etp_prior_gt.registration_head"],
        sys.modules["vlnce_baselines.models.etp_prior_gt.vilmodel_cmt"],
    )


def config(enabled=False, feedback=False):
    cfg = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_hidden_layers=1,
    )
    for key, value in dict(
        max_txt_task_embeddings=4,
        max_gmap_task_embeddings=3,
        max_action_steps=20,
        num_l_layers=1,
        num_x_layers=1,
        num_pano_layers=0,
        image_feat_size=8,
        depth_feat_size=4,
        angle_feat_size=4,
        use_depth_embedding=True,
        update_lang_bert=True,
        graph_sprels=True,
        use_lang2visn_attn=True,
        pred_head_dropout_prob=0.0,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
        navigation_architecture="try5",
        map_coordinate_fusion=True,
        registration_enabled=enabled,
        registration_feedback=feedback,
        registration_n_max=16,
    ).items():
        setattr(cfg, key, value)
    return cfg


def inputs():
    generator = torch.Generator().manual_seed(4)
    return dict(
        txt_embeds=torch.randn(1, 3, 16, generator=generator),
        txt_masks=torch.tensor([[True, True, False]]),
        gmap_vpids=[[None, "0", "g0", "g1"]],
        gmap_step_ids=torch.tensor([[0, 1, 0, 0]]),
        gmap_img_fts=torch.randn(1, 4, 16, generator=generator),
        gmap_pos_fts=torch.randn(1, 4, 7, generator=generator),
        gmap_masks=torch.tensor([[True, True, True, False]]),
        gmap_visited_masks=torch.tensor([[False, True, False, False]]),
        gmap_pair_dists=torch.zeros(1, 4, 4),
        gmap_task_embeddings=torch.ones(1, 4, dtype=torch.long),
        map_tokens=torch.randn(1, 101, 16, generator=generator),
        map_token_masks=torch.ones(1, 101, dtype=torch.bool),
        gmap_map_coords=torch.tensor([
            [[float("nan"), float("nan")], [2, 3], [3, 4], [5, 6]]
        ]),
        reg_route_counts=torch.tensor([3]),
    )


def test_head_shapes_padding_stop_ignore_and_feedback_width(monkeypatch):
    head_module, _ = load_modules(monkeypatch)
    head = head_module.RouteRegistrationHead(16, feedback=True)
    prediction = head(
        torch.randn(2, 4, 16),
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
        torch.tensor([3, 5]),
    )
    assert prediction["route_logits"].shape == (2, 4, 16)
    assert prediction["progress"].shape == (2, 4)
    assert bool(((prediction["progress"] >= 0) & (prediction["progress"] <= 1)).all())
    assert not bool(prediction["node_mask"][:, 0].any())
    assert torch.equal(
        prediction["route_logits"][0].softmax(-1)[:, 3:], torch.zeros(4, 13)
    )
    tokens, mask = head_module.append_registration_feedback(
        torch.randn(2, 101, 16), torch.ones(2, 101, dtype=torch.bool), prediction
    )
    assert tokens.shape == (2, 102, 16) and mask.shape == (2, 102)
    assert prediction["feedback_token"].shape == (2, 1, 16)


def test_loss_ignores_minus_one_stop_and_padding(monkeypatch):
    module, _ = load_modules(monkeypatch)
    head = module.RouteRegistrationHead(8)
    prediction = head(
        torch.randn(1, 4, 8),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([3]),
    )
    target = torch.tensor([[0, 2, -1, 0]])
    progress = torch.tensor([[float("nan"), 0.5, float("nan"), float("nan")]])
    loss = module.registration_loss(prediction, target, progress)
    expected = torch.nn.functional.cross_entropy(
        prediction["route_logits"][:, 1], torch.tensor([2])
    )
    expected += (prediction["progress"][0, 1] - 0.5).square()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert head.route.weight.grad is not None
    ignored = module.registration_loss(
        prediction, torch.full_like(target, -1), progress
    )
    assert torch.isfinite(ignored) and ignored.item() == 0


def test_disabled_and_zero_weight_head_reproduce_navigation_bit_for_bit(monkeypatch):
    module, model_module = load_modules(monkeypatch)
    torch.manual_seed(7)
    control = model_module.GlocalTextPathNavCMT(config()).eval()
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    supervised = model_module.GlocalTextPathNavCMT(config(enabled=True)).eval()
    assert torch.equal(control_rng, torch.get_rng_state())
    for name, value in control.state_dict().items():
        assert torch.equal(value, supervised.state_dict()[name]), name
    assert not any("registration_head" in name for name in control.state_dict())
    with torch.no_grad():
        # Use a map path that actually influences the control, rather than
        # relying on the template's initially zero residual projection.
        control.graph_map_attention.residual_projection.weight.normal_()
        supervised.graph_map_attention.load_state_dict(
            control.graph_map_attention.state_dict()
        )
    baseline = control.forward_navigation(**inputs())
    result = supervised.forward_navigation(**inputs())
    assert torch.equal(baseline["global_logits"], result["global_logits"])
    assert torch.equal(baseline["gmap_embeds"], result["gmap_embeds"])
    ce = torch.nn.functional.cross_entropy(result["global_logits"], torch.tensor([2]))
    aux = module.registration_loss(
        result["registration"], torch.tensor([[-1, 0, 1, -1]]), torch.zeros(1, 4)
    )
    assert torch.equal(ce, ce + 0.0 * aux)


def test_feedback_token_is_consumed_by_actual_navigation(monkeypatch):
    _, model_module = load_modules(monkeypatch)
    model = model_module.GlocalTextPathNavCMT(
        config(enabled=True, feedback=True)
    ).eval()
    with torch.no_grad():
        model.graph_map_attention.residual_projection.weight.normal_(std=0.2)
    token_counts = []
    handle = model.graph_map_attention.register_forward_pre_hook(
        lambda _module, args: token_counts.append(args[2].shape[1])
    )
    output = model.forward_navigation(**inputs())
    handle.remove()
    assert token_counts == [101, 102]
    output["global_logits"][torch.isfinite(output["global_logits"])].sum().backward()
    gradient = model.registration_head.feedback_projection.weight.grad
    assert gradient is not None and bool((gradient.abs() > 0).any())


def test_production_pretrain_loader_preserves_base_weights_and_rng(monkeypatch):
    load_modules(monkeypatch)
    name = "vlnce_baselines.models.etp_prior_gt.vlnbert_init"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vlnce_baselines/models/etp_prior_gt/vlnbert_init.py"
    )
    assert spec is not None and spec.loader is not None
    loader = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, loader)
    spec.loader.exec_module(loader)
    monkeypatch.setattr(
        PretrainedConfig,
        "from_pretrained",
        classmethod(lambda *_args, **_kwargs: config()),
    )

    def model_config(enabled):
        return types.SimpleNamespace(
            pretrained_path=None,
            use_depth_embedding=True,
            use_sprels=True,
            fix_lang_embedding=False,
            fix_pano_embedding=False,
            MAP_ENCODER=types.SimpleNamespace(
                enabled=True, architecture="try5", coordinate_fusion=True
            ),
            REGISTRATION=types.SimpleNamespace(
                enabled=enabled, feedback=False, n_max=16
            ),
        )

    torch.manual_seed(18)
    control = loader.get_vlnbert_models(model_config(False), dropout_rate=0.0)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(18)
    registered = loader.get_vlnbert_models(model_config(True), dropout_rate=0.0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert registered.registration_head is not None
    for key, value in control.state_dict().items():
        assert torch.equal(value, registered.state_dict()[key]), key


def test_registration_rejects_non_try5_architecture(monkeypatch):
    _, model_module = load_modules(monkeypatch)
    cfg = config(enabled=True)
    cfg.navigation_architecture = "current"
    cfg.map_coordinate_fusion = False
    with pytest.raises(ValueError, match="Try5"):
        model_module.GlocalTextPathNavCMT(cfg)
