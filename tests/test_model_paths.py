from pathlib import Path


def test_shared_model_paths_are_repo_relative_strings():
    import model_paths

    assert model_paths.CLIP_VIT_B32_MODEL == "data/models/ViT-B-32.pt"
    assert (
        model_paths.LLAMA_3_1_8B_INSTRUCT_MODEL == "data/models/Llama-3.1-8B-Instruct"
    )
    assert Path(model_paths.CLIP_VIT_B32_MODEL).is_file()


def test_llm_boxes_uses_shared_default_model_path():
    import model_paths
    from vlnce_baselines.models.etp_llm import llm_boxes_train

    args = llm_boxes_train.parse_args(["--gradient-checkpointing"])

    assert args.model_name_or_path == model_paths.LLAMA_3_1_8B_INSTRUCT_MODEL
