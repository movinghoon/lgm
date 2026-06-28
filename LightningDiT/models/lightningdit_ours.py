"""
Modified from https://github.com/hustvl/LightningDiT/blob/main/models/lightningdit.py

    - add support for 1D sequence
    - include samplers inside the model
    - slightly different cfg conditioning (conditioned on **all channels**)
"""

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transport import Sampler, create_transport

from .layers import (
    # Block,
    LabelEmbedder,
    ModulatedLinear,
    PatchEmbed,
    TimestepEmbedder,
    # Transformer,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    LayerScale,
    Mlp,
    SwiGLUFFN,
    modulate,
    checkpoint
)
from .model_utils import SIZE_DICT


logger = logging.getLogger("DeTok")


class Attention(nn.Module):
    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        kv_dim: int | None = None,
        is_cross_attn: bool = False,
        proj_bias: bool = True,
        force_causal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.num_heads = num_heads
        kv_dim = dim if kv_dim is None else kv_dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_cross_attn = is_cross_attn
        self.force_causal = force_causal

        self.fused_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.fused_attn and not Attention._logged:
            Attention._logged = True
            logger.info(f"[Attention]: Using {torch.__version__} Fused Attention, {force_causal=}")

        if is_cross_attn:
            self.c_q = nn.Linear(dim, dim, bias=qkv_bias)  # context to q
            self.c_kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)  # context to kv
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: Tensor, data: Tensor = None, attn_mask=None, rope=None, sizes=None) -> Tensor:
        # attn_mask: this is actually an bias term. 0 for visible, -inf for invisible
        bs, n_ctx, C = x.shape

        # Get q,k,v - either from cross attention or self attention
        if self.is_cross_attn:
            raise NotImplementedError
        else:
            qkv = self.qkv(x).reshape(bs, n_ctx, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(dim=0)
        # Apply norms and rotary embeddings
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q, k = rope(q), rope(k)

        # Handle kv caching for inference
        if self.kv_cache:
            if self.k_cache is None:
                k_cache, v_cache = k, v
            else:
                assert n_ctx in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)
            self.k_cache, self.v_cache = k_cache, v_cache
            k, v = k_cache, v_cache

        # Compute attention - use fused attention if available
        if self.fused_attn:
            assert attn_mask is None, "attn_mask disabled for merged attention"
            attn_mask = None
            if sizes is not None:
                attn_mask = sizes[None, None, None, :].float().log().to(x.device)

            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )    
        else:
            raise NotImplementedError

        # Project output
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(bs, n_ctx, C)))


class Block(nn.Module):
    _logged = False
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_bias: bool = True,
        attn_proj_bias: bool = True,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        kv_dim: int | None = None,
        use_cross_attn: bool = False,
        use_modulation: bool = False,
        use_swiglu: bool = False,
        force_causal: bool = False,
        no_dropout_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_modulation = use_modulation
        if not Block._logged:
            Block._logged = True
            logger.info(f"[Block]: {use_modulation=}, {use_cross_attn=}, {use_swiglu=}, {norm_layer=}")

        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            kv_dim=kv_dim if kv_dim is None else kv_dim,
            is_cross_attn=use_cross_attn,
            proj_bias=attn_proj_bias,
            force_causal=force_causal,
        )
        if self.use_cross_attn:
            self.data_norm = norm_layer(dim if kv_dim is None else kv_dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        if use_swiglu:
            self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                bias=mlp_bias,
                drop=proj_drop if not no_dropout_in_mlp else 0.0,
            )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.adaLN_modulation = None
        if self.use_modulation:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: Tensor,
        data: Tensor = None,
        attn_mask: Tensor = None,
        condition: Tensor = None,
        rope=None,
        sizes=None,
    ) -> Tensor:
        if self.use_modulation:
            assert condition is not None, "condition should not be None for modulation"
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                condition
            ).chunk(6, dim=-1)
            if condition.ndim == 2:  # (bsz, dim) -> (bsz, 1, dim)
                gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
                shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
                shift_mlp, scale_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa = None, None, 1.0
            shift_mlp, scale_mlp, gate_mlp = None, None, 1.0
        if self.use_cross_attn:
            raise NotImplementedError
        else:
            attn = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                attn_mask=attn_mask,
                rope=rope,
                sizes=sizes,
            )
        x = x + gate_msa * self.ls1(attn)
        x = x + gate_mlp * self.ls2(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        mlp_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_proj_bias: bool = True,
        init_values: float | None = None,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        block_fn: nn.Module = Block,
        use_swiglu: bool = False,
        force_causal: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    mlp_bias=mlp_bias,
                    attn_proj_bias=attn_proj_bias,
                    use_swiglu=use_swiglu,
                    force_causal=force_causal,
                )
                for _ in range(depth)
            ]
        )
        self.grad_checkpointing = grad_checkpointing
        logger.info(f"[Transformer]: grad_checkpointing={grad_checkpointing}")

    def forward(
        self,
        x: torch.Tensor,
        data: Tensor = None,
        attn_mask: Tensor = None,
        condition: Tensor = None,
        rope=None,
        sizes=None,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, data, attn_mask, condition, rope, sizes)
            else:
                x = block(x, data, attn_mask, condition, rope, sizes)
        return x


class LightningDiT(nn.Module):
    """lightning diffusion transformer model."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        num_sampling_steps=250,
        sampling_method="euler",
        timestep_shift=0.3,
        grad_checkpointing=False,
        force_one_d_seq=0,
        learn_sigma=False,  # no learn_sigma in SiT
        legacy_mode=False,
        qk_norm=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.token_channels = self.out_channels = token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.force_one_d_seq = force_one_d_seq
        self.grad_checkpointing = grad_checkpointing
        self.learn_sigma = learn_sigma
        self.legacy_mode = legacy_mode

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]
        self.width = width

        # --------------------------------------------------------------------------
        # embedding layers
        if self.force_one_d_seq > 0:
            self.x_embedder = nn.Linear(token_channels, width)
            # we use learnable positional embeddings for 1D sequence without rope
            self.pos_embed = nn.Parameter(torch.randn(1, self.force_one_d_seq, width) * 0.02)
            self.seq_len = self.force_one_d_seq
        else:
            self.x_embedder = PatchEmbed(self.input_size, patch_size, token_channels, width)
            # use rotary position encoding + abe, borrow from EVA
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, width))
            self.rope = VisionRotaryEmbeddingFast(width // num_heads // 2, self.input_size // patch_size)
            self.seq_len = num_patches

        self.t_embedder = TimestepEmbedder(width)
        self.y_embedder = LabelEmbedder(num_classes, width, label_drop_prob)

        # --------------------------------------------------------------------------
        # transformer architecture
        self.transformer = Transformer(
            width,
            num_layers,
            num_heads,
            block_fn=partial(Block, use_modulation=True),
            norm_layer=nn.RMSNorm,
            grad_checkpointing=grad_checkpointing,
            use_swiglu=True,
            qk_norm=qk_norm,
        )
        self.final_layer = ModulatedLinear(width, patch_size**2 * token_channels, use_rmsnorm=True)

        # --------------------------------------------------------------------------
        # transport and sampling setup
        self.transport = create_transport(use_cosine_loss=True, use_lognorm=True)
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=int(num_sampling_steps),
            timestep_shift=timestep_shift,
        )

        self.initialize_weights()

        # log model info
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[LightningDiT] params: {num_trainable_params:.2f}M size: {model_size}, num_layers: {num_layers}, width: {width}"
        )

    def initialize_weights(self):
        """initialize model weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize (and freeze) pos_embed by sin-cos embedding
        if not self.force_one_d_seq:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers in LightningDiT blocks
        for block in self.transformer.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """convert patch tokens back to image tensor."""
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def net(self, x, t=None, y=None, proj_mat=None, sizes=None):
        # ==============================================================================================
        pos_embed = self.pos_embed
        if proj_mat is not None and sizes is not None:
            assert x.shape[1] == proj_mat.shape[1], "should be already projected sample"

            # merge positional embeddings
            proj_mat = proj_mat.to(x.device)
            pos_embed = torch.einsum('BND,NM->BMD', pos_embed, proj_mat)
        
        if proj_mat is None and (x.shape[1] != pos_embed.shape[1]):
            num = x.shape[1]
            pos_embed = pos_embed[:, :num, :].contiguous()
        # ==============================================================================================
        
        """core network forward pass."""
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t) + self.y_embedder(y, self.training)  # (N, D)

        # check if self.pos_embed requires grad
        if not self.force_one_d_seq:
            if sizes is not None:
                raise NotImplementedError
            x = self.transformer(x, condition=c, rope=self.rope)  # (N, T, D)
        else:
            x = self.transformer(x, condition=c, sizes=sizes)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        if not self.force_one_d_seq:
            x = self.unpatchify(x)
        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale, proj_mat=None, sizes=None, cfg_interval=None, cfg_interval_start=None, cfg_interval_end=None, cfg_rescale_mode="none"):
        """forward pass with classifier-free guidance.

        cfg_rescale_mode:
          - "none": standard CFG, half_eps = uncond + cfg_scale * (cond - uncond)
          - "per_position": divide (cfg_scale - 1) by sizes[k] per position
            so the effective per-position cfg is 1 + (cfg_scale - 1) / s_k.
            Compensates for K-dependent score magnitude (|Δs| ∝ s_k) that
            otherwise makes the same cfg number do s_k× different work across K.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.net(combined, t, y, proj_mat=proj_mat, sizes=sizes)

        if self.legacy_mode:
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cat_dim = 1
        elif self.force_one_d_seq:
            # 1D path: model_out is (B, T, C), so slice the channel dim (last) — not
            # dim 1 (= T). The original `[:, :token_channels]` silently truncated the
            # sequence to token_channels tokens for force_one_d_seq, leaving the rest
            # untouched by CFG.
            eps, rest = model_out[..., : self.token_channels], model_out[..., self.token_channels :]
            cat_dim = -1
        else:
            eps, rest = model_out[:, : self.token_channels], model_out[:, self.token_channels :]
            cat_dim = 1

        if False:
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            diff = cond_eps - uncond_eps
            ones = torch.ones_like(diff) # B x N x C
            diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
            norm_threshold = 2.5
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor
            diff_parallel, diff_orthogonal = project(diff, cond_eps)
            normalized_update = diff_orthogonal
            half_eps = cond_eps + (cfg_scale - 1) * normalized_update
            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start or timestep > cfg_interval_end:
                    half_eps = cond_eps
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=cat_dim)
        else:
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            if cfg_rescale_mode == "per_position" and sizes is not None:
                sizes_factor = sizes.to(eps.device).float()[None, :, None]  # (1, K, 1)
                half_eps = cond_eps + (cfg_scale - 1) / sizes_factor * (cond_eps - uncond_eps)
            else:
                half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start or timestep > cfg_interval_end:
                    half_eps = cond_eps

            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=cat_dim)


    def _make_pooled_noise(self, shape, proj_mat, device):
        """Generate noise via pool_fn: eps_full ~ N(0,I_N) -> M @ eps_full ~ N(0, diag(1/|S_j|))"""
        B, K, D = shape
        N = proj_mat.shape[0]
        eps_full = torch.randn(B, N, D, device=device)
        return torch.einsum('bnd,nm->bmd', eps_full, proj_mat)

    def forward(self, x, y, proj_mat=None, sizes=None, loss_weight_mode="sizes", use_pooled_noise=True):
        """forward pass for training."""
        noise = None
        loss_weight = None
        if proj_mat is not None and sizes is not None:
            if use_pooled_noise:
                noise = self._make_pooled_noise(x.shape, proj_mat, x.device)
            if loss_weight_mode == "sizes":
                loss_weight = sizes[None, :, None].float().to(x.device)  # (1, K, 1)
                loss_weight = loss_weight * (sizes.shape[0] / sizes.sum())  # normalize so sum = K
            elif loss_weight_mode == "scalar":
                loss_weight = (sizes.shape[0] / sizes.sum()).float().to(x.device)

        loss_dict = self.transport.training_losses(
            self.net, x, dict(y=y, proj_mat=proj_mat, sizes=sizes),
            noise=noise, loss_weight=loss_weight,
        )
        return loss_dict["loss"].mean()

    @torch.inference_mode()
    def generate(self, n_samples, labels, proj_mat=None, sizes=None, cfg=1.0, args=None, cfg_interval_start=0.1, cfg_interval_end=1.0, cfg_rescale_mode="none", use_pooled_noise=True):
        """generate samples using the model."""
        device = labels.device

        # prepare noise tensor
        if self.force_one_d_seq:
            if proj_mat is not None and sizes is not None and use_pooled_noise:
                z = self._make_pooled_noise(
                    (n_samples, proj_mat.shape[1], self.token_channels), proj_mat, device)
            elif proj_mat is not None and sizes is not None:
                # legacy: model trained without pooled noise — keep K (=proj_mat.shape[1])
                z = torch.randn(n_samples, proj_mat.shape[1], self.token_channels)
            else:
                z = torch.randn(n_samples, self.force_one_d_seq, self.token_channels)
        else:
            z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size)
        z = z.to(device)

        # setup classifier-free guidance
        if cfg > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([self.num_classes] * n_samples, device=device)
            labels = torch.cat([labels, y_null], 0)
            model_kwargs = dict(y=labels,
                                cfg_scale=cfg,
                                cfg_interval=True,
                                cfg_interval_start=cfg_interval_start,
                                cfg_interval_end=cfg_interval_end,
                                proj_mat=proj_mat, sizes=sizes,
                                cfg_rescale_mode=cfg_rescale_mode)
            model_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels, proj_mat=proj_mat, sizes=sizes)
            model_fn = self.net

        # generate samples
        samples = self.sample_fn(z, model_fn, **model_kwargs)[-1]
        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples

        if proj_mat is not None and sizes is not None:  # unmerge
            if sizes.sum() == 256:
                un_proj_mat = proj_mat * sizes
                samples = torch.einsum('BMD,MN->BND', samples, un_proj_mat.T)
        return samples

    @torch.inference_mode()
    def generate2(self, n_samples, labels, proj_mat=None, sizes=None, cfg=1.0, args=None, cfg_rescale_mode="none"):
        """generate samples using the model."""
        device = labels.device

        # prepare noise tensor
        if self.force_one_d_seq:
            if proj_mat is not None and sizes is not None:
                z = self._make_pooled_noise(
                    (n_samples, proj_mat.shape[1], self.token_channels), proj_mat, device)
            else:
                z = torch.randn(n_samples, self.force_one_d_seq, self.token_channels)
        else:
            z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size)
        z = z.to(device)

        # setup classifier-free guidance
        if cfg > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([self.num_classes] * n_samples, device=device)
            labels = torch.cat([labels, y_null], 0)
            model_kwargs = dict(y=labels, cfg_scale=cfg, cfg_interval=True, cfg_interval_start=0.10, proj_mat=proj_mat, sizes=sizes, cfg_rescale_mode=cfg_rescale_mode)
            model_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels, proj_mat=proj_mat, sizes=sizes)
            model_fn = self.net

        # generate samples
        samples = self.sample_fn(z, model_fn, **model_kwargs)[-1]
        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples

        return samples

# model size variants
def LightningDiT_base(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="base", **kwargs)

def LightningDiT_large(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="large", **kwargs)


def LightningDiT_xl(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="xl", **kwargs)


def LightningDiT_huge(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="huge", **kwargs)

def LightningDiT_tmp(**kwargs) -> LightningDiT:
    return LightningDiT(model_size="tmp", **kwargs)


LightningDiT_ours_models = {
    "LightningDiT_ours_xl": LightningDiT_xl,
}
