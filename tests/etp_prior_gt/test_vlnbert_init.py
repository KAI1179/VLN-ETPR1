from types import SimpleNamespace


def test_vlnbert_init_passes_map_fusion_mode_to_visual_config(monkeypatch):
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp_prior_gt import vilmodel_cmt, vlnbert_init

    captured = {}

    monkeypatch.setattr(
        PretrainedConfig,
        "from_pretrained",
        staticmethod(lambda _name: SimpleNamespace()),
    )
    monkeypatch.setattr(
        vilmodel_cmt.GlocalTextPathNavCMT,
        "from_pretrained",
        classmethod(
            lambda cls, pretrained_model_name_or_path, config, state_dict: captured.setdefault(
                "config",
                config,
            )
        ),
    )

    model_config = SimpleNamespace(
        pretrained_path=None,
        use_depth_embedding=True,
        use_sprels=True,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
        MAP_ENCODER=SimpleNamespace(fusion="try5"),
    )

    vlnbert_init.get_vlnbert_models(model_config)

    assert captured["config"].map_fusion == "try5"
