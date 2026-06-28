import argparse
import itertools
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import lpips
from modelling.tokenizer import VQ_models
from utils.data import center_crop_arr

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


def main(args):
    assert torch.cuda.is_available(), "Tokenizer evaluation requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    tokenizer = build_tokenizer(model_args)
    tokenizer.load_state_dict(checkpoint["ema"])
    tokenizer = tokenizer.cuda().eval()

    lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).cuda().eval()

    run_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    if "outputs/" in args.ckpt_path and "/checkpoints" in args.ckpt_path:
        run_name = args.ckpt_path.split("outputs/")[-1].split("/checkpoints")[0]
    sample_folder_dir = os.path.join(args.sample_dir, f"{run_name}_{args.num}")
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    global_batch_size = args.per_proc_batch_size * world_size
    psnr_values = []
    ssim_values = []
    lpips_values = []
    fid_metric = FrechetInceptionDistance(normalize=False).cuda()
    loader = tqdm(loader) if rank == 0 else loader
    total = 0

    for x, _ in loader:
        if args.image_size_eval != args.image_size:
            gt = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        else:
            gt = x
        gt_numpy = (gt.permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0

        x = x.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            samples = tokenizer(x, args.num)[0]
            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")

        lpips_batch = lpips_model(samples.float(), gt.float())
        lpips_values.extend(lpips_batch.detach().cpu().numpy().flatten().tolist())

        samples_uint8 = torch.clamp(127.5 * samples + 128, 0, 255).to(dtype=torch.uint8)
        gt_uint8 = torch.clamp(127.5 * gt + 128, 0, 255).to(dtype=torch.uint8)
        fid_metric.update(gt_uint8, real=True)
        fid_metric.update(samples_uint8, real=False)

        samples_numpy = samples_uint8.permute(0, 2, 3, 1).cpu().numpy()
        for i, (sample, rgb_gt) in enumerate(zip(samples_numpy, gt_numpy)):
            index = i * world_size + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            rgb_restored = sample.astype(np.float32) / 255.0
            psnr_values.append(psnr_loss(rgb_gt, rgb_restored, data_range=1.0))
            ssim_values.append(ssim_loss(rgb_gt, rgb_restored, data_range=1.0, channel_axis=-1))

        total += global_batch_size

    fid = fid_metric.compute().detach()
    dist.barrier()

    gathered_psnr = [None for _ in range(world_size)]
    gathered_ssim = [None for _ in range(world_size)]
    gathered_lpips = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_psnr, psnr_values)
    dist.all_gather_object(gathered_ssim, ssim_values)
    dist.all_gather_object(gathered_lpips, lpips_values)

    if rank == 0:
        psnr_mean = np.mean(list(itertools.chain(*gathered_psnr)))
        ssim_mean = np.mean(list(itertools.chain(*gathered_ssim)))
        lpips_mean = np.mean(list(itertools.chain(*gathered_lpips)))
        print(f"PSNR: {psnr_mean:f}, SSIM: {ssim_mean:f}, LPIPS: {lpips_mean:f}")
        print(f"FID: {fid:f}")

        result_file = f"{sample_folder_dir}_results.txt"
        print(f"writing results to {result_file}")
        with open(result_file, "w", encoding="utf-8") as f:
            print(f"PSNR: {psnr_mean:f}, SSIM: {ssim_mean:f}, LPIPS: {lpips_mean:f}", file=f)
            print(f"FID: {fid:f}", file=f)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num", type=int, default=256)
    parser.add_argument("--data-path", type=str, default="/root/ImageNet/val")
    parser.add_argument("--dataset", type=str, choices=["imagenet"], default="imagenet")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lpips-net", type=str, choices=["alex", "vgg", "squeeze"], default="vgg")
    args = parser.parse_args()
    main(args)
