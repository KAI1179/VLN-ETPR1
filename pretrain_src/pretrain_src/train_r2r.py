import os
import time
from collections import defaultdict
from pathlib import Path
from easydict import EasyDict
from tqdm import tqdm

from no_tensorflow import configure_no_tensorflow

configure_no_tensorflow()

import torch
import torch.nn.functional as F

import torch.cuda.amp as amp  # TODO

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.models.optimizer_profiles import (
    ONLINE_FUSION_OPTIMIZER_PROFILE,
    configure_pretrain_optimizer_profile,
    optimizer_profile_summary,
)
from vlnce_baselines.models.etp_prior_gt.pretrain_checkpoint import (
    FUSION_SOURCE_PREFIX,
    MAP_ENCODER_SOURCE_PREFIX,
    load_checkpoint_submodule,
    validate_pretraining_initialization,
)

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.loader import MetaLoader, PrefetchLoader, build_dataloader
from data.dataset import R2RTextPathData
from data.tasks import MlmDataset, mlm_collate, SapDataset, sap_collate

from model.pretrain_cmt import GlocalTextPathCMTPreTraining
import numpy as np


def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    dataloaders = {}
    if is_train:
        tasks = data_cfg.tasks
    else:
        tasks = data_cfg.val_tasks
    for k, task_name in enumerate(tasks):
        if task_name == "mlm":
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == "sap":
            task_dataset = SapDataset(nav_db, tok, end_vp_pos_ratio=0.15)
            task_collate_fn = sap_collate
        else:
            raise ValueError(f"Undefined task {task_name}")

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = None
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def main(opts):
    default_gpu, n_gpu, device = set_cuda(opts)
    print(default_gpu, n_gpu, device)

    if default_gpu:
        LOGGER.info(
            "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16
            )
        )

    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, "logs"))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, "ckpts"))
        add_log_to_file(os.path.join(opts.output_dir, "logs", "log.txt"))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config["tasks"])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)
    candidate = (
        None
        if opts.cognitive_map_source is None
        else CognitiveMapCandidate.parse(
            opts.navigation_architecture,
            opts.cognitive_map_source,
        )
    )
    model_config.navigation_architecture = (
        "current" if candidate is None else candidate.architecture.value
    )
    model_config.cognitive_map_source = (
        None if candidate is None else candidate.source.value
    )
    model_config.map_loss_weight = getattr(opts, "map_loss_weight", 0.1)
    model_config.trajectory_keypoint_loss_weight = getattr(
        opts, "trajectory_keypoint_loss_weight", 0.001
    )
    model_config.map_predictor_checkpoint = getattr(
        opts, "map_predictor_checkpoint", ""
    )
    model_config.cognitive_map_namespace = opts.cognitive_map_namespace
    model_config.llm_cache_model_key = opts.llm_cache_model_key
    model_config.llm_cache_dir = opts.llm_cache_dir
    model_config.cognitive_map_target_namespace = (
        opts.cognitive_map_target_namespace
    )
    model_config.online_grid_loss_weight = opts.online_grid_loss_weight
    model_config.online_state_loss_weight = opts.online_state_loss_weight
    model_config.online_visual_loss_weight = opts.online_visual_loss_weight
    model_config.online_progress_loss_weight = opts.online_progress_loss_weight
    model_config.online_ghost_loss_weight = opts.online_ghost_loss_weight

    tokenizer = AutoTokenizer.from_pretrained("./bert_config/xlm-roberta-base")

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(
            opts.checkpoint, map_location=lambda storage, loc: storage
        )
        print(f"Loaded ckpt: {opts.checkpoint}")
    else:
        checkpoint = {}
        if opts.init_pretrained == "roberta":
            print("Init Model Name: ", model_config.lang_bert_name)
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
            for param_name, param in tmp.named_parameters():
                param_name = "bert." + param_name
                if "bert.encoder.layer" in param_name:
                    param_name = param_name.replace(
                        "bert.encoder.layer", "bert.lang_encoder.layer"
                    )
                checkpoint[param_name] = param
            # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
            checkpoint["bert.embeddings.token_type_embeddings.weight"] = torch.cat(
                [checkpoint["bert.embeddings.token_type_embeddings.weight"]] * 2, 0
            )
            del tmp
        elif opts.init_pretrained == "lxmert":
            tmp = torch.load(
                "pretrain_src/datasets/pretrained/LXMERT/model_LXRT.pth",
                map_location=lambda storage, loc: storage,
            )
            for param_name, param in tmp.items():
                param_name = param_name.replace("module.", "")
                if "bert.encoder.layer" in param_name:
                    param_name = param_name.replace(
                        "bert.encoder.layer", "bert.lang_encoder.layer"
                    )
                    checkpoint[param_name] = param
                elif "bert.encoder.x_layers" in param_name:
                    param_name = param_name.replace(
                        "bert.encoder.x_layers", "bert.global_encoder.encoder.x_layers"
                    )
                    checkpoint[param_name] = param
                    if "visual_attention" in param_name:
                        twin_param_name = param_name.replace(
                            "visual_attention", "text_attention"
                        )
                        checkpoint[twin_param_name] = param
                if "cls.predictions" in param_name:
                    param_name = param_name.replace(
                        "cls.predictions", "mlm_head.predictions"
                    )
                    checkpoint[param_name] = param
                else:
                    checkpoint[param_name] = param
            base = f"bert.global_encoder.encoder.x_layers.{model_config.num_x_layers - 1}.visn_output.LayerNorm"
            ln_weight_key = base + ".weight"
            ln_bias_key = base + ".bias"
            if ln_weight_key in checkpoint:
                checkpoint["graph_attentioned_txt_embeds_transform.norm.weight"] = (
                    checkpoint[ln_weight_key].clone()
                )
                print(
                    "Copy last visn_output.LayerNorm.weight to graph_attentioned_txt_embeds_transform.norm.weight"
                )
            else:
                print(f"No {ln_weight_key} found")
            if ln_bias_key in checkpoint:
                checkpoint["graph_attentioned_txt_embeds_transform.norm.bias"] = (
                    checkpoint[ln_bias_key].clone()
                )
                print(
                    "Copy last visn_output.LayerNorm.bias to graph_attentioned_txt_embeds_transform.norm.bias"
                )
            else:
                print(f"No {ln_bias_key} found")
            del tmp

    model_class = GlocalTextPathCMTPreTraining

    # update some training configs
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint
    )
    if (
        opts.checkpoint
        and opts.optimizer_profile == ONLINE_FUSION_OPTIMIZER_PROFILE
    ):
        checkpoint_path = Path(opts.checkpoint)
        validate_pretraining_initialization(
            model,
            checkpoint,
            checkpoint_path,
            allowed_missing_prefixes=(
                MAP_ENCODER_SOURCE_PREFIX,
                FUSION_SOURCE_PREFIX,
            ),
        )
        load_checkpoint_submodule(
            model.map_encoder,
            checkpoint,
            checkpoint_path=checkpoint_path,
            source_prefix=MAP_ENCODER_SOURCE_PREFIX,
            module_name="map_encoder",
            required=False,
        )
        load_checkpoint_submodule(
            model.bert.global_encoder.graph_map_attention,
            checkpoint,
            checkpoint_path=checkpoint_path,
            source_prefix=FUSION_SOURCE_PREFIX,
            module_name="graph_map_attention",
            required=False,
        )
    model.train()
    set_dropout(model, opts.dropout)  # 0.1
    configure_pretrain_optimizer_profile(
        model,
        opts.optimizer_profile,
    )
    model = wrap_model(model, device, opts.local_rank)
    lr_scales = configure_pretrain_optimizer_profile(
        model,
        opts.optimizer_profile,
    )
    if default_gpu:
        LOGGER.info(
            "Pretraining optimizer profile %s:\n%s",
            opts.optimizer_profile,
            optimizer_profile_summary(lr_scales, model.named_parameters()),
        )
    del checkpoint

    # load data training set
    data_cfg = EasyDict(opts.train_datasets["R2R"])
    map_dataset_kwargs = {
        "candidate": candidate,
        "cognitive_map_namespace": opts.cognitive_map_namespace,
        "llm_cache_dir": opts.llm_cache_dir,
        "llm_cache_model_key": opts.llm_cache_model_key,
        "cognitive_map_target_namespace": opts.cognitive_map_target_namespace,
        "visual_evidence_cache": opts.visual_evidence_cache,
    }
    train_nav_db = R2RTextPathData(
        data_cfg.train_traj_files,
        data_cfg.img_ft_file,
        data_cfg.dep_ft_file,
        data_cfg.scanvp_cands_file,
        data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        depth_feat_size=model_config.depth_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        val_sample_num=None,
        random_rotation_augmentation=False,
        **map_dataset_kwargs,
    )
    val_r2r_nav_db = R2RTextPathData(
        data_cfg.val_unseen_r2r_traj_files,
        data_cfg.img_ft_file,
        data_cfg.dep_ft_file,
        data_cfg.scanvp_cands_file,
        data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        depth_feat_size=model_config.depth_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        val_sample_num=opts.val_sample_num,
        **map_dataset_kwargs,
    )
    val_rxr_nav_db = R2RTextPathData(
        data_cfg.val_unseen_rxr_traj_files,
        data_cfg.img_ft_file,
        data_cfg.dep_ft_file,
        data_cfg.scanvp_cands_file,
        data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        depth_feat_size=model_config.depth_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len,
        in_memory=True,
        val_sample_num=opts.val_sample_num,
        **map_dataset_kwargs,
    )

    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, device, opts
    )
    val_r2r_dataloaders = create_dataloaders(
        data_cfg, val_r2r_nav_db, tokenizer, False, device, opts
    )
    val_rxr_dataloaders = create_dataloaders(
        data_cfg, val_rxr_nav_db, tokenizer, False, device, opts
    )

    meta_loader = MetaLoader(
        train_dataloaders,
        data_cfg.mix_ratio,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device,
    )

    meta_loader = PrefetchLoader(meta_loader, device)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts, lr_scales=lr_scales)

    if opts.fp16:
        grad_scaler = amp.GradScaler()

    global_step = 0
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info(
        "  Batch size = %d",
        opts.train_batch_size
        if opts.local_rank == -1
        else opts.train_batch_size * opts.world_size,
    )
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {
        task: RunningMeter(f"loss/{task}") for task in train_dataloaders.keys()
    }

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    mlm_loss = []
    sap_loss = []
    for step, (name, batch) in enumerate(meta_loader):
        n_examples[name] += batch["txt_ids"].size(0)
        n_in_units[name] += batch["txt_lens"].sum().item()
        task = name.split("_")[0]
        if opts.fp16:
            with amp.autocast():
                loss = model(batch, task=task, compute_loss=True)
        else:
            loss = model(batch, task=task, compute_loss=True)

        n_loss_units[name] += loss.size(0)
        loss = loss.mean()

        # backward pass
        if opts.gradient_accumulation_steps > 1:
            loss = loss / opts.gradient_accumulation_steps

        if opts.fp16:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        task2loss[name](loss.item())
        if name == "mlm":
            mlm_loss.append(loss.item())
        elif name == "sap":
            sap_loss.append(loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_this_step * param_group.get("lr_scale", 1.0)
            TB_LOGGER.add_scalar("lr", lr_this_step, global_step)

            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scalar_dict(
                {ll.name: ll.val for ll in task2loss.values() if ll.val is not None}
            )
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                if opts.fp16:
                    grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar("grad_norm", grad_norm, global_step)
            if opts.fp16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % opts.log_steps == 0:
                # monitor training throughput
                LOGGER.info(f"==============Step {global_step}===============")
                for t in train_dataloaders.keys():
                    tot_ex = n_examples[t]
                    ex_per_sec = int(tot_ex / (time.time() - start_time))
                    tot_in = n_in_units[t]
                    in_per_sec = int(tot_in / (time.time() - start_time))
                    tot_l = n_loss_units[t]
                    l_per_sec = int(tot_l / (time.time() - start_time))
                    LOGGER.info(f"{t}: {tot_ex} examples trained at {ex_per_sec} ex/s")
                    TB_LOGGER.add_scalar(f"perf/{t}_ex_per_s", ex_per_sec, global_step)
                    TB_LOGGER.add_scalar(f"perf/{t}_in_per_s", in_per_sec, global_step)
                    TB_LOGGER.add_scalar(f"perf/{t}_loss_per_s", l_per_sec, global_step)
                LOGGER.info("===============================================")
                LOGGER.info(
                    f"MLM mean loss during this log interval: {np.mean(mlm_loss)}"
                )
                LOGGER.info(
                    f"SAP mean loss during this log interval: {np.mean(sap_loss)}"
                )
                mlm_loss = []
                sap_loss = []

            if global_step % opts.valid_steps == 0:
                LOGGER.info(
                    f"------Step {global_step}: start validation R2R unseen------"
                )
                validate(model, val_r2r_dataloaders, setname="_unseen")
                LOGGER.info(
                    f"------Step {global_step}: start validation RxR unseen------"
                )
                validate(model, val_rxr_dataloaders, setname="_unseen")
                model_saver.save(model, global_step)
        if global_step >= opts.num_train_steps:
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f"------Step {global_step}: start validation R2R unseen------")
        validate(model, val_r2r_dataloaders, setname="_unseen")
        LOGGER.info(f"------Step {global_step}: start validation RxR unseen------")
        validate(model, val_rxr_dataloaders, setname="_unseen")
        model_saver.save(model, global_step)


def validate(model, val_dataloaders, setname=""):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        if task.startswith("mlm"):
            val_log = validate_mlm(model, loader)
        elif task.startswith("sap"):
            val_log = validate_sap(model, loader)
        else:
            raise ValueError(f"Undefined task {task}")
        val_log = {f"val{setname}_{task}_{k}": v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f"valid{setname}_{task}/{k}": v for k, v in val_log.items()}
        )
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task="mlm", compute_loss=False)
        labels = batch["txt_labels"]
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction="sum")
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_word = sum(all_gather(n_word))
    tot_time = time.time() - st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {"loss": val_loss, "acc": acc, "tok_per_s": n_word / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, acc: {acc * 100:.2f}")
    return val_log


def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]
    n_correct = (outputs == labels).sum().item()
    return n_correct


@torch.no_grad()
def validate_sap(model, val_loader):
    LOGGER.info("start running SAP validation...")
    val_gloss = 0
    n_gcorrect = 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        global_logits, global_act_labels = model(batch, task="sap", compute_loss=False)
        val_gloss += F.cross_entropy(
            global_logits, global_act_labels, reduction="sum"
        ).data.item()
        n_gcorrect += torch.sum(
            torch.argmax(global_logits, 1) == global_act_labels
        ).item()
        n_data += len(global_act_labels)

    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data

    tot_time = time.time() - st
    val_log = {"gloss": val_gloss, "gacc": gacc, "tok_per_s": n_data / tot_time}
    LOGGER.info(
        f"validation finished in {int(tot_time)} seconds, gacc: {gacc * 100:.2f}"
    )
    return val_log


def build_args():
    parser = load_parser()

    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts


if __name__ == "__main__":
    args = build_args()
    main(args)
