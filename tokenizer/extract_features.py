import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from sklearn.cluster import AgglomerativeClustering

from datasets.imagenet import get_imagenet_dataloader
from datasets.img_latent_dataset import ImgLatentDataset
from modelling.tokenizer import VQ_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _getattr(args, name, default):
    return getattr(args, name, default)


def build_tokenizer(model_args):
    tokenizer = VQ_models[model_args.vq_model](
        image_size=model_args.image_size,
        codebook_size=model_args.codebook_size,
        codebook_embed_dim=model_args.codebook_embed_dim,
        codebook_l2_norm=model_args.codebook_l2_norm,
        commit_loss_beta=model_args.commit_loss_beta,
        entropy_loss_ratio=model_args.entropy_loss_ratio,
        vq_loss_ratio=model_args.vq_loss_ratio,
        kl_loss_weight=model_args.kl_loss_weight,
        dropout_p=model_args.dropout_p,
        enc_type=model_args.enc_type,
        encoder_model=model_args.encoder_model,
        dec_type=model_args.dec_type,
        decoder_model=model_args.decoder_model,
        num_latent_tokens=model_args.num_latent_tokens,
        enc_tuning_method=model_args.encoder_tuning_method,
        dec_tuning_method=model_args.decoder_tuning_method,
        enc_pretrained=model_args.encoder_pretrained,
        dec_pretrained=model_args.decoder_pretrained,
        enc_patch_size=model_args.encoder_patch_size,
        dec_patch_size=model_args.decoder_patch_size,
        tau=model_args.tau,
        repa=model_args.repa,
        repa_model=model_args.repa_model,
        repa_patch_size=model_args.repa_patch_size,
        repa_proj_dim=model_args.repa_proj_dim,
        repa_loss_weight=model_args.repa_loss_weight,
        repa_align=model_args.repa_align,
        num_codebooks=model_args.num_codebooks,
        enc_token_drop=model_args.enc_token_drop,
        enc_token_drop_max=model_args.enc_token_drop_max,
        aux_dec_model=model_args.aux_decoder_model,
        aux_loss_mask=model_args.aux_loss_mask,
        aux_hog_dec=model_args.aux_hog_decoder,
        aux_dino_dec=model_args.aux_dino_decoder,
        aux_clip_dec=model_args.aux_clip_decoder,
        aux_supcls_dec=model_args.aux_supcls_decoder,
        to_pixel=model_args.to_pixel,
        variable_method=model_args.variable_method,
        merge_dim=model_args.merge_dim,
        vf_loss_weight=_getattr(model_args, "vf_loss_weight", 1.0),
        vf_margin=_getattr(model_args, "vf_margin", 0.25),
    )
    return tokenizer


def save_latent_shard(output_dir, rank, shard_idx, latents, latents_flip, labels):
    latents = torch.cat(latents, dim=0)
    latents_flip = torch.cat(latents_flip, dim=0)
    labels = torch.cat(labels, dim=0)
    save_dict = {
        "latents": latents,
        "latents_flip": latents_flip,
        "labels": labels,
    }
    save_filename = os.path.join(output_dir, f"latents_rank{rank:02d}_shard{shard_idx:03d}.safetensors")
    save_file(
        save_dict,
        save_filename,
        metadata={
            "total_size": f"{latents.shape[0]}",
            "dtype": f"{latents.dtype}",
            "device": f"{latents.device}",
        },
    )
    return save_filename


def save_merge_assignments(tokenizer, model_args, output_path, name):
    embeddings = tokenizer.merge_module.embeddings.detach()
    max_len = model_args.num_latent_tokens
    block_mask = torch.block_diag(
        torch.ones(max_len // 4, max_len // 4),
        torch.ones(max_len // 4, max_len // 4),
        torch.ones(max_len // 4, max_len // 4),
        torch.ones(max_len // 4, max_len // 4),
    ).to(embeddings.device)

    with torch.no_grad():
        metric = F.normalize(embeddings, dim=-1).float()
        scores = metric @ metric.T
        scores = scores * block_mask + (1 - block_mask) * -100.0
        scores = (1 - scores).cpu().numpy()

    merge_dict = {}
    for num in [4, 8, 16, 32, 48, 64, 96, 128, 196, 256]:
        if num > max_len:
            continue
        clustering = AgglomerativeClustering(
            n_clusters=num,
            metric="precomputed",
            linkage="average",
            distance_threshold=None,
        )
        labels = clustering.fit(scores).labels_
        assignment = F.one_hot(torch.tensor(labels), num_classes=num).to(metric.device).float()
        proj_mat = assignment / assignment.sum(dim=0, keepdim=True)
        sizes = assignment.sum(dim=0)
        merge_dict[num] = (proj_mat.cpu(), sizes.cpu())

    save_path = os.path.join(output_path, f"{name}.pt")
    torch.save(merge_dict, save_path)
    print(f"Saved merge assignments to {save_path}")


def main(args):
    accelerator = Accelerator()
    rank = accelerator.process_index
    device = accelerator.device
    set_seed(args.seed + rank)

    output_dir = os.path.join(args.output_path, args.name, f"{args.data_split}_{args.image_size}")
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    if model_args.variable_method != "ours2":
        raise ValueError(f"extract_features.py only supports variable_method='ours2', got {model_args.variable_method!r}")

    tokenizer = build_tokenizer(model_args)
    tokenizer.load_state_dict(checkpoint["ema"])
    tokenizer = tokenizer.to(device).eval()

    loaders = [
        get_imagenet_dataloader(
            args.data_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            transform_mean=[0.5, 0.5, 0.5],
            transform_std=[0.5, 0.5, 0.5],
            random_flip_prob=0.0,
        ),
        get_imagenet_dataloader(
            args.data_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            transform_mean=[0.5, 0.5, 0.5],
            transform_std=[0.5, 0.5, 0.5],
            random_flip_prob=1.0,
        ),
    ]
    loaders = [accelerator.prepare(loader) for loader in loaders]

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    shard_batches = max(1, args.shard_size // args.batch_size)

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0] * accelerator.num_processes
        if run_images % 1000 == 0 and rank == 0:
            print(f"{datetime.now()} processing {run_images}")

        for loader_idx, data in enumerate(batch_data):
            x = data[0].to(device, non_blocking=True)
            y = data[1]
            with torch.no_grad():
                z = tokenizer.encode(x, num=256)[0].detach().cpu()

            if batch_idx == 0 and rank == 0:
                print("latent shape", z.shape, "dtype", z.dtype)

            if loader_idx == 0:
                latents.append(z)
                labels.append(y.cpu())
            else:
                latents_flip.append(z)

        if len(latents) >= shard_batches:
            save_filename = save_latent_shard(output_dir, rank, saved_files, latents, latents_flip, labels)
            if rank == 0:
                print(f"Saved {save_filename}")
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    if latents:
        save_filename = save_latent_shard(output_dir, rank, saved_files, latents, latents_flip, labels)
        if rank == 0:
            print(f"Saved {save_filename}")

    accelerator.wait_for_everyone()
    if rank == 0:
        ImgLatentDataset(output_dir, latent_norm=True)
        save_merge_assignments(tokenizer, model_args, args.output_path, args.name)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/root/ImageNet/train")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="latents")
    parser.add_argument("--name", type=str, default="ours2-d32")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
