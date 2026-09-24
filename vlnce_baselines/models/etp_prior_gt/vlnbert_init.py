import torch

PRETRAIN_FUSION_PREFIX = "bert.global_encoder.graph_map_attention."
NAV_FUSION_PREFIX = "bert.graph_map_attention."


def get_tokenizer(args):
    from transformers import AutoTokenizer

    if args.dataset == "rxr" or args.tokenizer == "xlm":
        cfg_name = "bert_config/xlm-roberta-base"
    else:
        cfg_name = "bert_config/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer


def get_vlnbert_models(config=None, dropout_rate=0.1):
    if config is None:
        raise ValueError("config is required")

    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp_prior_gt.vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    load_map_modules = bool(
        getattr(config.MAP_ENCODER, "load_pretrained_map_modules", True)
    )
    new_ckpt_weights = {}
    remapped_fusion = {}
    keywords = [
        "graph_query_text",
        "graph_attentioned_txt_embeds_transform",
        "global_sap_head",
    ]
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location="cpu")
        for k, v in ckpt_weights.items():
            if k.startswith("module"):
                new_ckpt_weights[k[7:]] = v
            if any(key in k for key in keywords):
                new_ckpt_weights["bert." + k] = v
            elif load_map_modules and k.startswith(PRETRAIN_FUSION_PREFIX):
                # Pretraining stores the fusion inside GlobalMapEncoder; the
                # navigation model owns it at the top level. Without this remap
                # the pretrained fusion is silently dropped as "unused".
                nav_key = NAV_FUSION_PREFIX + k[len(PRETRAIN_FUSION_PREFIX) :]
                new_ckpt_weights[nav_key] = v
                remapped_fusion[nav_key[len("bert.") :]] = v
            else:
                new_ckpt_weights[k] = v

    cfg_name = "bert_config/xlm-roberta-base"
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = "global"

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3
    vis_config.navigation_architecture = config.MAP_ENCODER.architecture

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights,
    )
    _verify_fusion_transfer(visual_model, remapped_fusion, model_name_or_path)
    return visual_model


def _verify_fusion_transfer(visual_model, remapped_fusion, model_name_or_path):
    """Fail loudly if remapped pretrained fusion tensors did not land."""
    if not remapped_fusion:
        print(
            "  graph_map_attention: no pretrained tensors loaded "
            "(checkpoint has none or load_pretrained_map_modules=False); "
            "fusion starts from random attention + zero residual projection"
        )
        return
    fusion_state = visual_model.graph_map_attention.state_dict()
    mismatched = [
        key
        for key, value in remapped_fusion.items()
        if key not in fusion_state
        or fusion_state[key].shape != value.shape
        or not torch.equal(fusion_state[key].cpu(), value.cpu())
    ]
    if mismatched:
        raise RuntimeError(
            "Pretrained graph_map_attention tensors were remapped but did not "
            f"land in the navigation model: {mismatched}"
        )
    print(
        f"  graph_map_attention: loaded {len(remapped_fusion)} pretrained tensors "
        f"from {model_name_or_path}"
    )
