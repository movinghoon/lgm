"""Evaluate a LoRA-tuned XL LightningDiT latent model."""

from peft import LoraConfig, get_peft_model
import argparse
import datetime
import logging
import os
import re
import sys
import time

import torch
import torch.distributed

import models
import utils.distributed as distributed
from utils.builders import create_generation_model, create_optimizer_and_scaler, create_train_dataloader
from utils.misc import ckpt_resume, save_checkpoint
from utils.path_config import path_value
from utils.train_utils import (
    collect_tokenizer_stats,
    evaluate_generator,
    setup,
    train_one_epoch_generator2,
    visualize_generator,
    visualize_tokenizer,
)
from datasets import ImgLatentDataset
import utils.distributed as distributed

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("DeTok")


def get_latent_dataloader(data_dir, batch_size, num_workers, shuffle=True):
    dataset = ImgLatentDataset(data_dir)
    mean, std = dataset._latent_mean, dataset._latent_std
    sampler_train = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_global_rank(),
        shuffle=True,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader_train, mean, std


def set_timestep_shift(model, shift, num_sampling_steps, sampling_method="euler"):
    """Rebuild model.sample_fn with a new timestep_shift."""
    model.sample_fn = model.sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=int(num_sampling_steps),
        timestep_shift=float(shift),
    )


def parse_last_fid_is(log_dir):
    """Read the last (fid, is) values written by evaluate_generator into eval_summary.txt."""
    summary_path = os.path.join(log_dir, "eval_summary.txt")
    if not os.path.exists(summary_path):
        return float("inf"), float("-inf")
    with open(summary_path, "r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return float("inf"), float("-inf")
    fid_m = re.search(r"fid=([0-9.]+)", lines[-1])
    is_m = re.search(r"is=([0-9.]+)", lines[-1])
    fid = float(fid_m.group(1)) if fid_m else float("inf")
    is_score = float(is_m.group(1)) if is_m else float("-inf")
    return fid, is_score


def main(args: argparse.Namespace) -> int:
    global logger
    wandb_logger = setup(args)

    # LoRA was trained legacy → plain noise + cfg_rescale_mode=none at sampling.
    # Override via --no_legacy_sampling if you really want pooled noise.
    if not getattr(args, 'no_legacy_sampling', False):
        args.legacy_sampling = True
        logger.info("legacy_sampling=True (LoRA-eval default): _make_pooled_noise disabled, "
                    "cfg_rescale_mode forced to 'none'")

    data_loader_train, mean, std = get_latent_dataloader(args.latent_data_path, args.batch_size, 4, shuffle=True)
    args.channel_dim = mean.shape[-1]

    # initialize models
    model, tokenizer, ema_model = create_generation_model(args)
    optimizer, loss_scaler = create_optimizer_and_scaler(args, model)

    # Optionally load base pretrained weights. This is redundant when the LoRA
    # checkpoint stores the full PEFT state dict (base + adapters), since that is
    # loaded strict below and overwrites everything — so it's optional.
    if args.pretrained_ckpt_path:
        base_ckpt = torch.load(args.pretrained_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(base_ckpt["model_ema"])
        logger.info(f"Loaded base pretrained weights from {args.pretrained_ckpt_path}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "attn.qkv", "attn.proj",
            "mlp.w12", "mlp.w3",
        ],
        lora_dropout=0.,
    )
    model = get_peft_model(model, lora_config, adapter_name='256')
    lora_ckpt = torch.load(args.lora_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(lora_ckpt["model"])
    logger.info(f"Loaded LoRA-tuned weights from {args.lora_ckpt_path}")
    ema_model = None
    model_wo_ddp = model

    # reset stats
    tokenizer.reset_stats(mean, std)

    # mapper --num_tokens_list
    if not args.merge_dict_path:
        raise ValueError("--merge_dict_path is required for refined LightningDiT LoRA evaluation")
    my_dict = torch.load(args.merge_dict_path)

    # setup distributed training
    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_wo_ddp = model.module

    # NOTE: we already loaded both base + LoRA above; do not call ckpt_resume.

    epoch = args.start_epoch - 1

    # build grid: (cfg, shift, cfg_interval_start, cfg_interval_end)
    cfg_list = args.cfg_list if args.cfg_list is not None else [args.cfg]
    shift_list = args.shift_list if args.shift_list is not None else [args.timestep_shift]
    cfg_int_start_list = (
        args.cfg_interval_start_list
        if args.cfg_interval_start_list is not None
        else [args.cfg_interval_start]
    )
    cfg_int_end_list = (
        args.cfg_interval_end_list
        if args.cfg_interval_end_list is not None
        else [args.cfg_interval_end]
    )

    # default best to the first (only) entry — covers singleton case where the
    # search loop is skipped, so the final 50k eval still honors the *_list /
    # singleton args rather than falling back to the scalar args.* values.
    best_cfg = cfg_list[0]
    best_shift = shift_list[0]
    best_cfg_int_start = cfg_int_start_list[0]
    best_cfg_int_end = cfg_int_end_list[0]

    total_combos = len(cfg_list) * len(shift_list) * len(cfg_int_start_list) * len(cfg_int_end_list)

    # only run search loop if more than one candidate
    if total_combos > 1:
        # ensure ascending cfg order so early-stopping logic on monotonic FID rise is meaningful
        cfg_list = sorted(cfg_list)

        fid_dict = {}
        stop_signal = torch.zeros(1, dtype=torch.int32, device='cuda')

        for cfg_int_start in cfg_int_start_list:
            for cfg_int_end in cfg_int_end_list:
                # write onto args so train_utils.generate_images picks these up
                args.cfg_interval_start = cfg_int_start
                args.cfg_interval_end = cfg_int_end

                for shift in shift_list:
                    set_timestep_shift(model_wo_ddp, shift, args.num_sampling_steps)
                    args.timestep_shift = shift  # so eval_summary logs the right shift

                    running_min_fid = float("inf")
                    fid_above_count = 0  # consecutive cfgs with fid > running_min + tolerance

                    for cfg in cfg_list:
                        evaluate_generator(
                            args, model_wo_ddp, ema_model, tokenizer, epoch, wandb_logger,
                            use_ema=False, cfg=cfg, num_images=args.num_images_for_eval_and_search,
                            my_dict=my_dict, num=args.num_tokens,
                        )
                        torch.distributed.barrier()

                        # rank 0 parses metrics, decides whether to stop, broadcasts the decision
                        stop_signal.zero_()
                        if distributed.is_main_process():
                            fid, is_score = parse_last_fid_is(args.log_dir)
                            fid_dict[(cfg, shift, cfg_int_start, cfg_int_end)] = fid
                            logger.info(
                                f"[search] cfg={cfg}, timestep_shift={shift}, "
                                f"cfg_interval=[{cfg_int_start},{cfg_int_end}], "
                                f"fid={fid}, is={is_score}"
                            )

                            if fid < running_min_fid:
                                running_min_fid = fid
                                fid_above_count = 0
                            elif fid > running_min_fid + args.fid_tolerance:
                                fid_above_count += 1
                            else:
                                fid_above_count = 0  # within tolerance: reset

                            stop = False
                            reason = None
                            if is_score >= args.is_stop:
                                stop, reason = True, f"is={is_score:.2f} >= is_stop={args.is_stop}"
                            elif fid_above_count >= args.fid_patience:
                                stop, reason = True, (
                                    f"fid above running_min({running_min_fid:.4f})+tol({args.fid_tolerance}) "
                                    f"for {fid_above_count} consecutive cfgs"
                                )

                            if stop:
                                logger.info(
                                    f"[search] EARLY STOP at cfg={cfg}, shift={shift}, "
                                    f"cfg_interval=[{cfg_int_start},{cfg_int_end}]: {reason}"
                                )
                                stop_signal[0] = 1

                        torch.distributed.broadcast(stop_signal, src=0)
                        if stop_signal.item() == 1:
                            break

        # find best (cfg, shift, cfg_int_start, cfg_int_end) on rank 0
        if distributed.is_main_process():
            best_fid = float("inf")
            for (cfg, shift, cfg_int_start, cfg_int_end), fid in fid_dict.items():
                if fid < best_fid:
                    best_fid = fid
                    best_cfg = cfg
                    best_shift = shift
                    best_cfg_int_start = cfg_int_start
                    best_cfg_int_end = cfg_int_end
            logger.info(
                f"Best FID: {best_fid}, Best cfg: {best_cfg}, "
                f"Best timestep_shift: {best_shift}, "
                f"Best cfg_interval: [{best_cfg_int_start},{best_cfg_int_end}]"
            )

    torch.distributed.barrier()

    # broadcast best (cfg, shift, cfg_int_start, cfg_int_end) to all ranks
    best_tensor = torch.tensor(
        [best_cfg, best_shift, best_cfg_int_start, best_cfg_int_end],
        dtype=torch.float32, device='cuda',
    )
    torch.distributed.broadcast(best_tensor, src=0)
    best_cfg = best_tensor[0].item()
    best_shift = best_tensor[1].item()
    best_cfg_int_start = best_tensor[2].item()
    best_cfg_int_end = best_tensor[3].item()
    torch.distributed.barrier()

    # final 50k eval with the best combo
    set_timestep_shift(model_wo_ddp, best_shift, args.num_sampling_steps)
    args.timestep_shift = best_shift
    args.cfg_interval_start = best_cfg_int_start
    args.cfg_interval_end = best_cfg_int_end

    evaluate_generator(
        args, model_wo_ddp, ema_model, tokenizer, epoch, wandb_logger,
        use_ema=False, cfg=best_cfg, num_images=args.num_images,
        my_dict=my_dict, num=args.num_tokens,
    )
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Generation model training", add_help=False)

    parser.add_argument("--num_tokens", default=32, type=int, help="number of tokens")

    # basic training parameters
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model", default="LightningDiT_ours_xl", type=str)
    parser.add_argument("--order", default="raster", type=str)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--no_dropout_in_mlp", action="store_true")
    parser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_one_d_seq", type=int, default=256, help="1d token sequence length")
    parser.add_argument("--legacy_mode", action="store_true")

    # tokenizer parameters
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--tokenizer", default="ours2_32", type=str)
    parser.add_argument("--token_channels", default=64, type=int)
    parser.add_argument("--tokenizer_patch_size", default=16, type=int)
    parser.add_argument("--use_ema_tokenizer", action=argparse.BooleanOptionalAction, default=True)

    # tokenizer cache parameters
    parser.add_argument("--collect_tokenizer_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer_bsz", default=256, type=int)
    parser.add_argument("--cached_path", type=str, default="data/imagenet_tokens/")
    parser.add_argument("--stats_key", type=str, default="ours2_32")
    parser.add_argument("--overwrite_stats", action="store_true")
    parser.add_argument("--stats_cache_path", type=str, default=path_value("paths", "stats_cache"))

    # logging parameters
    parser.add_argument("--output_dir", default=path_value("paths", "work_dir", "./work_dirs"))
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--vis_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)

    # checkpoint parameters
    parser.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_from", default=None, help="resume model weights and optimizer state")
    parser.add_argument("--load_from", type=str, default=None, help="load from pretrained model")
    parser.add_argument("--load_tokenizer_from", type=str, default=path_value("paths", "tokenizer_ckpt"), help="load from pretrained tokenizer")
    parser.add_argument("--keep_n_ckpts", default=1, type=int, help="keep the last n checkpoints")
    parser.add_argument("--milestone_interval", default=100, type=int, help="keep checkpoints every n epochs")

    # evaluation parameters
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--fid_stats_path", type=str, default=path_value("paths", "fid_ref_256"))
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # optimization parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.25, help="warmup_ep = warmup_rate * total_ep")
    parser.add_argument("--ema_rate", default=0.9999, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_aligned_schedule", action="store_true")

    # generation parameters
    parser.add_argument("--num_iter", default=64, type=int, help="number of autoregressive steps for MAR")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="noise schedule for diffusion")
    parser.add_argument("--cfg", default=1.6, type=float, help="cfg value for diffusion")
    parser.add_argument("--cfg_schedule", default="linear", type=str, help="cfg schedule for diffusion")
    parser.add_argument("--cfg_list", default=None, type=float, nargs="+", help="cfg list for search")
    parser.add_argument("--cfg_interval_start", default=0.15, type=float, help="cfg interval start (timestep fraction)")
    parser.add_argument("--cfg_interval_end", default=1.0, type=float, help="cfg interval end (timestep fraction)")
    parser.add_argument("--cfg_interval_start_list", default=None, type=float, nargs="+",
                        help="cfg_interval_start list for grid search; combined with cfg_list/shift_list/cfg_interval_end_list")
    parser.add_argument("--cfg_interval_end_list", default=None, type=float, nargs="+",
                        help="cfg_interval_end list for grid search; combined with cfg_list/shift_list/cfg_interval_start_list")
    parser.add_argument("--shift_list", default=None, type=float, nargs="+",
                        help="timestep_shift list for grid search; combined with cfg_list")
    parser.add_argument("--cfg_rescale_mode", default="none", type=str, choices=["none", "per_position"],
                        help="CFG rescaling: 'none' = standard CFG; "
                             "'per_position' = divide (cfg-1) by sizes[k] per position to "
                             "compensate for K-dependent score magnitude.")

    # cfg-search early stopping (per shift x cfg_interval combo)
    parser.add_argument("--is_stop", type=float, default=350.0,
                        help="stop sweeping cfg (this shift/cfg_interval) once IS reaches this value")
    parser.add_argument("--fid_patience", type=int, default=3,
                        help="stop after this many consecutive cfgs with FID > running_min + fid_tolerance")
    parser.add_argument("--fid_tolerance", type=float, default=0.1,
                        help="tolerance over running-min FID before counting toward fid_patience")

    # mar parameters
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--mask_ratio_min", type=float, default=0.7)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--buffer_size", type=int, default=64)

    # diffusion loss parameters
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=1024)
    parser.add_argument("--num_sampling_steps", type=str, default="250")
    parser.add_argument("--timestep_shift", type=float, default=1.1)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--temperature", default=1.0, type=float)

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true",
                        help="generate images of only the class of interest for args.num_images images")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=1, type=int)

    # wandb parameters
    parser.add_argument("--project", default="DIT", type=str)
    parser.add_argument("--entity", default=path_value("wandb", "entity", "YOUR_WANDB_ENTITY"), type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    parser.add_argument("--latent_data_path", default=path_value("paths", "latent_data"), type=str)
    parser.add_argument("--merge_dict_path", required=True, type=str,
                        help="path to the merge dict (.pt) used for num_tokens projection")

    # LoRA-eval-specific args
    parser.add_argument("--pretrained_ckpt_path", default=None, type=str,
                        help="optional base model checkpoint (with 'model_ema'); redundant when the "
                             "LoRA ckpt holds the full PEFT state dict, so it can be omitted")
    parser.add_argument("--lora_ckpt_path", required=True, type=str,
                        help="path to the LoRA-tuned checkpoint (with 'model' state_dict on the peft-wrapped model)")
    parser.add_argument("--lora_r", default=32, type=int, help="LoRA rank (must match training)")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA alpha (must match training)")
    parser.add_argument("--no_legacy_sampling", action="store_true",
                        help="opt out of the LoRA-eval default legacy_sampling=True; use pooled noise / per_position rescale instead")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
