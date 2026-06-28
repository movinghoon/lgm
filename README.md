
# Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers
This repository contains official code for \
[Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers](https://arxiv.org/abs/2606.20076) \
[Dong Hoon Lee](https://movinghoon.github.io) and [Seunghoon Hong](https://maga33.github.io) \
ICML 2026

## Tokenizer

Run from `tokenizer/`.

### Training

```bash
torchrun --nproc_per_node 8 train_tokenizer.py --config config.yaml
```

### Evaluation

Reconstruction metrics (PSNR, SSIM, LPIPS, FID):

```bash
torchrun --nproc_per_node 8 evaluate_tokenizer.py \
  --ckpt_path tokenizer.pt \
  --num 256 \
  --data-path /root/ImageNet/val \
  --sample-dir reconstructions
```

Extract latents + merge assignments (produces the `latent_data`):

```bash
accelerate launch --num_processes 8 extract_features.py \
  --ckpt_path tokenizer.pt \
  --data_path /root/ImageNet/train \
  --output_path latents \
  --name ours2-d32
```

## Generator (LightningDiT)

Run from `LightningDiT/`.

### Training

```bash
torchrun --nproc_per_node=8 train.py \
  --exp_name <name> \
  --epochs 400 \
  --use_aligned_schedule \
  --cfg_list 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
  --num_tokens_list 128 128 128 128 64 64 32 \
  --enable_wandb
```

### Evaluation example

```bash
torchrun --nproc_per_node=8 evaluate.py \
  --exp_name evaluate_xl_base \
  --resume_from ./xl.pth \
  --num_tokens 32 \
  --cfg_rescale_mode per_position \
  --num_sampling_steps 25 \
  --cfg 10.5

torchrun --nproc_per_node=8 evaluate.py \
  --exp_name evaluate_xl_base \
  --resume_from ./xl.pth \
  --num_tokens 64 \
  --cfg_rescale_mode per_position \
  --num_sampling_steps 25 \
  --cfg 8.0

torchrun --nproc_per_node=8 evaluate.py \
  --exp_name evaluate_xl_base \
  --resume_from ./xl.pth \
  --num_tokens 128 \
  --cfg_rescale_mode per_position \
  --num_sampling_steps 25 \
  --cfg 6.5
```

### (Optional) LoRA training

Length-specific fine-tuning of a base checkpoint. 

```bash
torchrun --nproc_per_node=8 train_lora.py \
  --exp_name <name> \
  --epochs 10 \
  --num_tokens_list <num_tokens> \
  --enable_wandb \
  --pretrained_ckpt_path xl.pth \
  --merge_dict_path path/to/ours2-d32.pt
```

### (Optional) LoRA evaluation example

```bash
torchrun --nproc_per_node=8 evaluate_lora.py \
  --exp_name evaluate_xl_lora \
  --num_tokens 32 \
  --lora_ckpt_path lora_32.pth \
  --merge_dict_path path/to/ours2-d32.pt \
  --num_sampling_steps 25 \
  --cfg 2.0
```

## Pretrained Checkpoints

| Resource | Description | Link |
|---|---|---|
| `latent_data` | `ours2-d32` tokenizer latents (dataset) | [link](https://huggingface.co/datasets/movinghoon/ours2-d32) |
| `tokenizer` | `ours2_32` tokenizer | [link](https://huggingface.co/movinghoon/LGM/blob/main/tokenizer.pt) |
| `xl_corrected` | Base XL generator, noise-scale corrected | [link](https://huggingface.co/movinghoon/LGM/blob/main/xl.pth) |
| `lora_32` | Length-specific LoRA for 32 tokens (incl. base params) | [link](https://huggingface.co/movinghoon/LGM/blob/main/lora_32.pth) |
| `lora_128` | Length-specific LoRA for 128 tokens (incl. base params) | [link](https://huggingface.co/movinghoon/LGM/blob/main/lora_128.pth) |


## Noise-Scale Correction (post-acceptance improvement)

After the paper was accepted, we discovered that matching noise statistics at each token
count `K` can greatly improve the joint-diffusion training quality, thereby removing the
necessity of length-specific LoRA fine-tuning. In detail, the generator is trained jointly
on the full token view (`z ∈ R^{N×D}`) and a merged view (`Mz ∈ R^{K×D}`), where `M` is
an average-pooling assignment matrix that reduces `N` tokens to `K`. The intuition behind
the improvement is that the merged view is *not* an independent diffusion process: if it
comes from a single physical process in the full space, its noise must be the projection
`Mε` (per-cluster variance `1/|S_j|`), not the isotropic `N(0, I_K)` we originally sampled.
Sampling isotropic noise overestimates the merged-view variance by the cluster size
`|S_j|`, so the diffusion target seen at `K` tokens disagrees with the one seen at `N`. We
fix this by drawing a single shared full-space noise `ε ~ N(0, I_N)`, projecting it through
`M` for the merged view, and reweighting the merged-view loss by `|S_j|` (normalized so the
loss scale stays invariant to the merge ratio); the same projected-noise structure is used
at sampling time. The LoRA workflow below is therefore kept only for legacy/base
checkpoints trained with plain noise, where `legacy_training` / `legacy_sampling` default
to `True`.

## Acknowledgements

This repository builds upon the following projects:

- [Continuous_tokenizer](https://github.com/Hhhhhhao/continuous_tokenizer)
- [DeTok](https://github.com/Jiawei-Yang/DeTok)

## Citation

If you find this work helpful, please cite:

```bibtex
@misc{lee2026lgm,
      title={Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers}, 
      author={Dong Hoon Lee and Seunghoon Hong},
      year={2026},
      url={https://arxiv.org/abs/2606.20076}, 
}
```

## Contact

For any inquiries, please contact **Dong Hoon Lee** at `donghoonlee.ai@gmail.com`.
