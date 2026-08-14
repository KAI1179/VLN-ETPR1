from pathlib import Path

from vlnce_baselines.models.etp_imagined.checkpoint import load_complete_state_dict
from vlnce_baselines.models.etp_prior_gt.pretrain_checkpoint import (
    FUSION_SOURCE_PREFIX,
    load_checkpoint_submodule,
    load_pretraining_checkpoint,
    map_pretraining_state_to_vlnbert,
)


def get_tokenizer(args):
    from transformers import AutoTokenizer

    if args.dataset == "rxr" or args.tokenizer == "xlm":
        cfg_name = "bert_config/xlm-roberta-base"
    else:
        cfg_name = "bert_config/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer


def get_vlnbert_models(
    config=None,
    dropout_rate=0.1,
    checkpoint_state=None,
):
    if config is None:
        raise ValueError("config is required")

    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp_prior_gt.vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        if checkpoint_state is None:
            checkpoint_state = load_pretraining_checkpoint(model_name_or_path)
        new_ckpt_weights = map_pretraining_state_to_vlnbert(checkpoint_state)

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
    if model_name_or_path is not None:
        map_cfg = config.MAP_ENCODER
        require_complete = bool(
            getattr(map_cfg, "require_complete_pretrained_modules", False)
        )
        if require_complete:
            runtime_state = {
                key[len("bert.") :]: value
                for key, value in new_ckpt_weights.items()
                if key.startswith("bert.")
            }
            load_complete_state_dict(
                visual_model,
                runtime_state,
                Path(model_name_or_path),
                "vln_bert",
            )
        load_checkpoint_submodule(
            visual_model.graph_map_attention,
            checkpoint_state,
            checkpoint_path=Path(model_name_or_path),
            source_prefix=FUSION_SOURCE_PREFIX,
            module_name="graph_map_attention",
            required=require_complete,
        )
    return visual_model
