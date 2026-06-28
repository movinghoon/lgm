import sys
from pathlib import Path

import torch
import torch.nn as nn


# The tokenizer package uses absolute `modelling.*` imports internally, so its
# `tokenizer/` dir must be on sys.path as the parent of the top-level `modelling` package.
# Append (don't insert at 0): both repos ship a `utils` package, and LightningDiT's own
# `utils` must keep priority so `utils.misc.NativeScalerWithGradNormCount` resolves here.
_TOKENIZER_ROOT = Path(__file__).resolve().parents[2] / "tokenizer"
if str(_TOKENIZER_ROOT) not in sys.path:
    sys.path.append(str(_TOKENIZER_ROOT))

from modelling.tokenizer import VQ_models  # noqa: E402


class ContinuousTokenizer(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_args = tmp["args"]
        state_dict = tmp["ema"]
        self.tokenizer = VQ_models[model_args.vq_model](
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
        )
        self.tokenizer.load_state_dict(state_dict)
        self.tokenizer = self.tokenizer.cuda().eval()
        self.args = model_args
        self.mean = 0.0
        self.std = 1.0

    def reset_stats(self, mean, std):
        self.mean = mean.cuda()
        self.std = std.cuda()

    def detokenize(self, latents):
        latents = latents * self.std + self.mean
        out = self.tokenizer.decode(latents)
        return (out * 0.5 + 0.5).clamp(0, 1)

    def tokenize(self, images, sampling=None):
        return (self.tokenizer.encode(images, num=256)[0] - self.mean) / self.std


def ours2_32(load_ckpt=True, load_from=None, gamma=0.0):
    if load_from is None:
        raise ValueError("ours2_32 requires --load_tokenizer_from or paths.tokenizer_ckpt in path.yaml")
    return ContinuousTokenizer(ckpt_path=load_from)


OUR_models = {
    "ours2_32": ours2_32,
}
