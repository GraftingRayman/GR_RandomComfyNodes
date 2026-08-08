"""
GR Unified Sage Attention Patch  (standalone -- no KJNodes dependency)
------------------------------------------------------------------------
One node covering 4 sage-attention patch strategies:

  - generic     : patches Comfy's shared optimized_attention() dispatcher.
                  Works on basically any model. Simplest, but doesn't get the
                  extra VRAM-saving tricks below.
  - ltx2        : block-level patch on LTX2's attn1 (to_q/to_k/to_v split),
                  with fused Triton RoPE+quant option.
  - wan         : block-level patch on Wan's self_attn + cross_attn
                  (T2V/I2V variants handled separately).
  - minimax_h3  : block-level patch on MiniMax H3's fused qkv_proj, with
                  optional per-head-group chunking to shrink the quantized
                  working set.

The block-level paths share one kernel dispatcher (`_sageattn_int8_fp8_nhd`)
that picks the right sageattention CUDA-arch kernel (sm80/86/89/90/120/121)
and does int8/fp8 quantization with explicit tensor `del`s to keep peak VRAM
down -- they only differ in how q/k/v are pulled out of each model's blocks.

`architecture=auto` detects LTX2 / Wan / MiniMax H3 from the model structure
and falls back to `generic` for anything else, so it's a safe default.

Dependencies: torch, sageattention (required for anything but you not using
sage at all), triton (optional -- only used for the fused RoPE kernel on
LTX2/Wan when triton_kernels=True; falls back to eager RoPE without it).
No dependency on ComfyUI-KJNodes.

Drop into your GR node package and register in NODE_CLASS_MAPPINGS /
NODE_DISPLAY_NAME_MAPPINGS as usual.
"""

import logging
import importlib
import types

import torch

import comfy.model_management as mm
import comfy.ldm.modules.attention as _comfy_attn
from comfy.ldm.modules.attention import wrap_attn, attention_pytorch, optimized_attention

try:
    from comfy.ldm.lightricks.model import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

try:
    from comfy.ldm.lightricks.model import (
        GuideAttentionMask as _GuideAttentionMask,
        _attention_with_guide_mask as _ltx_attn_with_guide_mask,
    )
except ImportError:
    _GuideAttentionMask = None
    _ltx_attn_with_guide_mask = None

try:
    from comfy.ldm.flux.math import apply_rope as _wan_apply_rope
except ImportError:
    _wan_apply_rope = None

try:
    from comfy.ldm.wan.model import WanT2VCrossAttention as _WanT2VCrossAttention, WanI2VCrossAttention as _WanI2VCrossAttention
except ImportError:
    _WanT2VCrossAttention = _WanI2VCrossAttention = None

try:
    from comfy.ldm.minimax.model import MiniMaxH3Model as _MiniMaxH3Model
    from comfy.quant_ops import ck as _ck
except ImportError:
    _MiniMaxH3Model = None
    _ck = None

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# generic path: sageattn_modes + get_sage_func (patches optimized_attention)
# ---------------------------------------------------------------------------

sageattn_modes = [
    "disabled", "auto",
    "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3", "sageattn3_per_block_mean",
]


def get_sage_func(sage_attention, allow_compile=False):
    logging.info(f"GRUnifiedSageAttentionPatch: using sage attention mode: {sage_attention}")
    if sage_attention == "auto":
        from sageattention import sageattn
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell
        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(q, k, v, is_causal=is_causal, attn_mask=attn_mask, per_block_mean=(sage_attention == "sageattn3_per_block_mean"))
            return out.transpose(1, 2) if tensor_layout == "NHD" else out
    else:
        raise ValueError(f"Unknown sage_attention mode: {sage_attention}")

    if not allow_compile:
        sage_func = torch.compiler.disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(lambda t: t.view(b, -1, heads, dim_head), (q, k, v))
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out
    return attention_sage


# ---------------------------------------------------------------------------
# CUDA arch / sageattention kernel resolution (shared by the block-level paths)
# ---------------------------------------------------------------------------

def get_cuda_version():
    try:
        version = torch.version.cuda
        if version is not None:
            major, minor = version.split('.')
            return int(major), int(minor)
        return 0, 0
    except Exception:
        return 0, 0


sageplus_sm89_available = False
_cuda_archs = None
try:
    from sageattention.core import (
        per_thread_int8_triton, per_warp_int8_cuda, per_block_int8_triton,
        per_channel_fp8, get_cuda_arch_versions, attn_false,
    )
    _cuda_archs = get_cuda_arch_versions()
except Exception:
    pass

_QATTN_PROBE = {
    "sm80": "qk_int8_sv_f16_accum_f32_attn",
    "sm89": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    "sm90": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
}


def _resolve_qattn(arch):
    try:
        core = importlib.import_module("sageattention.core")
    except Exception:
        return None
    candidates = [getattr(core, f"_qattn_{arch}", None), getattr(core, f"{arch}_compile", None)]
    try:
        mod = importlib.import_module(f"sageattention.{arch}_compile")
        candidates += [mod, getattr(mod, f"_qattn_{arch}", None)]
    except Exception:
        pass
    for obj in candidates:
        if obj is not None and hasattr(obj, _QATTN_PROBE[arch]):
            return obj
    return None


_qattn_sm80 = _resolve_qattn("sm80")
_qattn_sm89 = _resolve_qattn("sm89")
_qattn_sm90 = _resolve_qattn("sm90")

if _qattn_sm89 is not None:
    sageplus_sm89_available = hasattr(_qattn_sm89, 'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf') and get_cuda_version() >= (12, 8)


# ---------------------------------------------------------------------------
# Triton kernels: fused RoPE (q/k) and per-thread int8 quant (int64-safe)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _rope_qk_split_kernel(
        Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
        H, T, D, TH,
        IS_BF16: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        bht = tl.program_id(0)
        t = bht % T
        h = (bht // T) % H
        b = bht // TH

        D_half = D // 2
        cols = tl.arange(0, BLOCK_HD)
        mask = cols < D_half

        qk_base = (b * T + t) * (H * D) + h * D
        cs_base = (b * H * T + h * T + t) * D_half

        q_x = tl.load(Q_ptr + qk_base + cols, mask=mask, other=0.0).to(tl.float32)
        q_y = tl.load(Q_ptr + qk_base + D_half + cols, mask=mask, other=0.0).to(tl.float32)
        k_x = tl.load(K_ptr + qk_base + cols, mask=mask, other=0.0).to(tl.float32)
        k_y = tl.load(K_ptr + qk_base + D_half + cols, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(Cos_ptr + cs_base + cols, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(Sin_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)

        q_ox = q_x * cos - q_y * sin
        q_oy = q_y * cos + q_x * sin
        k_ox = k_x * cos - k_y * sin
        k_oy = k_y * cos + k_x * sin

        if IS_BF16:
            tl.store(Q_ptr + qk_base + cols, q_ox.to(tl.bfloat16), mask=mask)
            tl.store(Q_ptr + qk_base + D_half + cols, q_oy.to(tl.bfloat16), mask=mask)
            tl.store(K_ptr + qk_base + cols, k_ox.to(tl.bfloat16), mask=mask)
            tl.store(K_ptr + qk_base + D_half + cols, k_oy.to(tl.bfloat16), mask=mask)
        else:
            tl.store(Q_ptr + qk_base + cols, q_ox.to(tl.float16), mask=mask)
            tl.store(Q_ptr + qk_base + D_half + cols, q_oy.to(tl.float16), mask=mask)
            tl.store(K_ptr + qk_base + cols, k_ox.to(tl.float16), mask=mask)
            tl.store(K_ptr + qk_base + D_half + cols, k_oy.to(tl.float16), mask=mask)

    @triton.jit
    def _quant_query_per_thread_int8_i64_kernel(Input, Output, Scale, L,
                                                stride_iz, stride_ih, stride_in,
                                                stride_oz, stride_oh, stride_on,
                                                stride_sz, stride_sh,
                                                C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
        offs_k = tl.arange(0, C)

        input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

        x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
        x = x.to(tl.float32)
        scale = tl.max(tl.abs(x)) / 127. + 0.0000001
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        x_int8 = x_int8.to(tl.int8)
        tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
        tl.store(scale_ptrs, scale)

    @triton.jit
    def _quant_key_per_thread_int8_i64_kernel(Input, Output, Scale, L,
                                              stride_iz, stride_ih, stride_in,
                                              stride_oz, stride_oh, stride_on,
                                              stride_sz, stride_sh,
                                              C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)

        input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
        x0 = x0.to(tl.float32)
        x1 = x1.to(tl.float32)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        x0_int8 = x0_int8.to(tl.int8)
        x1_int8 = x1_int8.to(tl.int8)
        tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
        tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
        tl.store(scale_ptrs, scale)


def fused_rope_qk(q, k, freqs_cis, use_triton=True):
    """Apply split RoPE to q and k in one fused kernel pass.
    q, k: [B, T, H*D] contiguous. freqs_cis: (cos, sin, split_pe), cos/sin: [B, H, T, D//2].
    Falls back to comfy's apply_rotary_emb if preconditions or triton aren't available.
    """
    if not (use_triton and HAS_TRITON and q.is_cuda) or apply_rotary_emb is None:
        return apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)

    cos, sin = freqs_cis[0], freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else False

    if not split_pe or cos.ndim != 4 or q.ndim != 3:
        return apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)

    B_cos, H, T_cos, D_half = cos.shape
    D = D_half * 2
    if q.shape != (B_cos, T_cos, H * D) or k.shape != (B_cos, T_cos, H * D):
        return apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)

    q = q.contiguous()
    k = k.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()

    BLOCK_HD = triton.next_power_of_2(D_half)
    num_warps = min(max(BLOCK_HD // 32, 1), 8)
    _rope_qk_split_kernel[(B_cos * H * T_cos,)](
        q, k, cos_c, sin_c, H, T_cos, D, T_cos * H,
        IS_BF16=(q.dtype == torch.bfloat16), BLOCK_HD=BLOCK_HD, num_warps=num_warps,
    )
    return q, k


def _per_thread_int8_i64(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, tensor_layout="NHD"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    _quant_query_per_thread_int8_i64_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    _quant_key_per_thread_int8_i64_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK,
    )

    return q_int8, q_scale, k_int8, k_scale


def _sageattn_int8_fp8_nhd(qkv, dtype):
    """qkv: [q, k, v] each [batch, seq_len, num_heads, head_dim] NHD layout. List is
    consumed so `del` frees the float tensors as soon as each arch-branch is done with
    them -- attention is the VRAM peak in these models."""
    q, k, v = qkv
    qkv.clear()
    head_dim_og = q.shape[-1]

    tensor_layout = "NHD"
    _tensor_layout = 0
    _is_caual = 0
    _qk_quant_gran = 3
    _return_lse = 0
    sm_scale = head_dim_og ** -0.5
    quant_v_scale_max = 448.0

    if _cuda_archs[0] in {"sm80", "sm86"}:
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        v_fp16 = v.to(torch.float16)
        del v
        _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v_fp16, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif _cuda_archs[0] == "sm75":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout)
        del q, k
        o, _ = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=None, return_lse=False)
        del v
    elif _cuda_archs[0] == "sm89":
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif _cuda_archs[0] == "sm90":
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
        del q, k
        seq_dim = 1
        kv_len = v.size(seq_dim)
        v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
        if v_pad_len > 0:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=seq_dim)
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif _cuda_archs[0] in {"sm120", "sm121"}:
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        _qk_quant_gran = 2
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    else:
        raise RuntimeError(f"GRUnifiedSageAttentionPatch: unsupported CUDA architecture '{_cuda_archs[0]}'")

    del q_int8, q_scale, k_int8, k_scale
    return o


# ---------------------------------------------------------------------------
# Per-architecture block forwards
# ---------------------------------------------------------------------------

def ltx2_sageattn_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    dtype = x.dtype
    context = x if context is None else context

    q = self.to_q(x)
    q = self.q_norm(q)
    k = self.to_k(context)
    k = self.k_norm(k)
    if pe is not None:
        use_triton = getattr(self, 'use_triton_kernels', False)
        if k_pe is None:
            q, k = fused_rope_qk(q, k, pe, use_triton=use_triton)
        else:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, k_pe)
    v = self.to_v(context)

    if mask is not None:
        if _GuideAttentionMask is not None and isinstance(mask, _GuideAttentionMask):
            o = _ltx_attn_with_guide_mask(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        else:
            o = _comfy_attn.optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            _b, _t, _ = o.shape
            o = o.view(_b, _t, self.heads, self.dim_head)
            o.mul_((2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1))
            o = o.view(_b, _t, self.heads * self.dim_head)
            del gate_logits
        return self.to_out(o)

    batch_size, seq_len, _ = q.shape
    head_dim_og = self.dim_head

    q = q.view(batch_size, seq_len, self.heads, head_dim_og)
    k = k.view(batch_size, k.shape[1], self.heads, head_dim_og)
    v = v.view(batch_size, v.shape[1], self.heads, head_dim_og)

    qkv = [q, k, v]
    del q, k, v
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        o.mul_((2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1))
        del gate_logits

    return self.to_out(o.view(batch_size, seq_len, -1))


def wan_sageattn_forward(self, x, freqs, transformer_options={}):
    dtype = x.dtype
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    q, k = _wan_apply_rope(q, k, freqs)
    v = self.v(x).view(b, s, n, d)

    qkv = [q, k, v]
    del q, k, v
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

    return self.o(o.view(b, s, n * d))


def wan_t2v_cross_sageattn_forward(self, x, context, transformer_options={}, **kwargs):
    dtype = x.dtype
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)

    qkv = [q, k, v]
    del q, k, v
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

    return self.o(o.view(b, s, n * d))


def wan_i2v_cross_sageattn_forward(self, x, context, context_img_len, transformer_options={}):
    dtype = x.dtype
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    context_img = context[:, :context_img_len]
    context = context[:, context_img_len:]

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
    v_img = self.v_img(context_img).view(b, -1, n, d)
    qkv_img = [q, k_img, v_img]
    del k_img, v_img
    img_o = _sageattn_int8_fp8_nhd(qkv_img, dtype)

    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)
    qkv = [q, k, v]
    del q, k, v
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

    o.add_(img_o)
    del img_o
    return self.o(o.view(b, s, n * d))


def minimax_sageattn_forward(self, x, rope_freqs=None, transformer_options={}):
    if isinstance(x, list):
        x = x.pop()
    dtype = x.dtype
    device = x.device
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    del x
    q = q.view(1, s, self.heads, self.head_dim)
    k = k.view(1, s, self.heads, self.head_dim)
    v = v.view(1, s, self.heads, self.head_dim)
    if rope_freqs is not None:
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        _ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)

    n = min(transformer_options.get("minimax_head_chunks", 1), self.heads) if isinstance(transformer_options, dict) else 1
    if n <= 1:
        qkv = [q, k, v]
        del q, k, v
        o = _sageattn_int8_fp8_nhd(qkv, dtype)
        return self.out_proj(o.view(s, self.heads * self.head_dim))

    out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
    out_nhd = out.view(1, s, self.heads, self.head_dim)
    hs = 0
    for i in range(n):
        he = hs + self.heads // n + (1 if i < self.heads % n else 0)
        out_nhd[:, :, hs:he] = _sageattn_int8_fp8_nhd([q[:, :, hs:he], k[:, :, hs:he], v[:, :, hs:he]], dtype)
        hs = he
    del q, k, v
    return self.out_proj(out)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class GRUnifiedSageAttentionPatch:
    """
    One node for all sage-attention patch flavors: 'generic' (works on any model,
    patches the shared attention dispatcher), or the lower-peak-VRAM block-level
    patches for 'ltx2' / 'wan' / 'minimax_h3'. 'auto' detects the architecture and
    falls back to 'generic' for anything else. No dependency on ComfyUI-KJNodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "architecture": (
                    ["auto", "generic", "ltx2", "wan", "minimax_h3"],
                    {"default": "auto", "tooltip": "auto-detect from diffusion_model (falls back to 'generic' if unrecognized), or force a specific patch path."},
                ),
            },
            "optional": {
                "sage_attention_mode": (
                    sageattn_modes,
                    {"default": "auto", "tooltip": "generic path only: which sageattn kernel variant to use."},
                ),
                "allow_compile": ("BOOLEAN", {"default": False, "tooltip": "generic path only: allow torch.compile to trace into the sage attention function."}),
                "triton_kernels": ("BOOLEAN", {"default": True, "tooltip": "ltx2 only: use fused Triton RoPE kernel on self-attn Q/K (requires triton, falls back to eager if unavailable)."}),
                "minimax_head_chunks": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "minimax_h3 only: split heads into N groups to lower peak VRAM."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "GRNodes/experimental"
    DESCRIPTION = (
        "Unified sage attention patch (standalone, no KJNodes dependency). Covers "
        "the generic optimized_attention_override path plus LTX2 / Wan / MiniMax H3 "
        "block-level patches in one node, with auto-detection of which applies."
    )
    EXPERIMENTAL = True

    @staticmethod
    def _detect_architecture(diffusion_model):
        if hasattr(diffusion_model, "blocks"):
            blocks = diffusion_model.blocks
            if len(blocks) and hasattr(blocks[0], "attn") and hasattr(blocks[0].attn, "qkv_proj"):
                return "minimax_h3"
            if len(blocks) and hasattr(blocks[0], "self_attn"):
                return "wan"

        if hasattr(diffusion_model, "transformer_blocks"):
            tb = diffusion_model.transformer_blocks
            if len(tb) and hasattr(tb[0], "attn1") and hasattr(tb[0].attn1, "to_q"):
                return "ltx2"

        return "generic"

    def patch(self, model, architecture, sage_attention_mode="auto", allow_compile=False,
              triton_kernels=True, minimax_head_chunks=1):
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        if architecture == "auto":
            architecture = self._detect_architecture(diffusion_model)
            logging.info(f"GRUnifiedSageAttentionPatch: auto-detected architecture '{architecture}'")
        else:
            logging.info(f"GRUnifiedSageAttentionPatch: applying '{architecture}' sage attention patch")

        if architecture == "generic":
            if sage_attention_mode == "disabled":
                return (model_clone,)
            new_attention = get_sage_func(sage_attention_mode, allow_compile=allow_compile)

            def attention_override_sage(func, *args, **kwargs):
                return new_attention.__wrapped__(*args, **kwargs)

            model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
            return (model_clone,)

        if _cuda_archs is None:
            raise RuntimeError(
                "sageattention is not installed / not new enough / CUDA architecture "
                "could not be determined -- cannot apply a block-level sage attention "
                "patch. Try architecture='generic' instead."
            )

        if architecture == "ltx2":
            if apply_rotary_emb is None:
                raise RuntimeError("Could not import apply_rotary_emb from comfy.ldm.lightricks.model -- cannot patch LTX2.")
            for idx, block in enumerate(diffusion_model.transformer_blocks):
                block.attn1.use_triton_kernels = triton_kernels
                model_clone.add_object_patch(
                    f"diffusion_model.transformer_blocks.{idx}.attn1.forward",
                    ltx2_sageattn_forward.__get__(block.attn1, block.attn1.__class__),
                )

        elif architecture == "wan":
            if _wan_apply_rope is None:
                raise RuntimeError("Could not import apply_rope from comfy.ldm.flux.math -- cannot patch Wan.")
            for idx, block in enumerate(diffusion_model.blocks):
                model_clone.add_object_patch(
                    f"diffusion_model.blocks.{idx}.self_attn.forward",
                    wan_sageattn_forward.__get__(block.self_attn, block.self_attn.__class__),
                )
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is not None and _WanI2VCrossAttention is not None and type(cross_attn) is _WanI2VCrossAttention:
                    model_clone.add_object_patch(
                        f"diffusion_model.blocks.{idx}.cross_attn.forward",
                        wan_i2v_cross_sageattn_forward.__get__(cross_attn, cross_attn.__class__),
                    )
                elif cross_attn is not None and _WanT2VCrossAttention is not None and type(cross_attn) is _WanT2VCrossAttention:
                    model_clone.add_object_patch(
                        f"diffusion_model.blocks.{idx}.cross_attn.forward",
                        wan_t2v_cross_sageattn_forward.__get__(cross_attn, cross_attn.__class__),
                    )

        elif architecture == "minimax_h3":
            if _ck is None:
                raise RuntimeError("This ComfyUI version does not support MiniMax H3 (comfy.quant_ops.ck missing).")
            if _MiniMaxH3Model is not None and not isinstance(diffusion_model, _MiniMaxH3Model):
                raise RuntimeError("architecture='minimax_h3' but diffusion_model is not a MiniMaxH3Model instance.")
            transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
            transformer_options["minimax_head_chunks"] = minimax_head_chunks
            model_clone.model_options["transformer_options"] = transformer_options
            for idx, block in enumerate(diffusion_model.blocks):
                model_clone.add_object_patch(
                    f"diffusion_model.blocks.{idx}.attn.forward",
                    minimax_sageattn_forward.__get__(block.attn, block.attn.__class__),
                )

        else:
            raise ValueError(f"Unknown architecture '{architecture}'")

        return (model_clone,)


try:
    # Sol-Attn's actual kernels (_tri_fwd/_int8_fwd/_morton*) are proprietary Triton
    # code we haven't seen -- we call kijai's real node rather than reimplement it.
    solattn_mod = importlib.import_module("custom_nodes.ComfyUI-SolAttn_triton")
except ImportError:
    try:
        solattn_mod = importlib.import_module("ComfyUI-SolAttn_triton")
    except ImportError:
        solattn_mod = None


class GRUnifiedAccelerator:
    """
    One node, three independent on/off systems: sage attention (generic or
    block-level ltx2/wan/minimax_h3), Sol-Attn sparse routing, and a reminder to
    chain Spectrum's own node afterward (its forecasting internals are genuinely
    per-architecture and not folded in here -- see node description).

    IMPORTANT COMPATIBILITY NOTE: Sol-Attn only intercepts the 'generic' sage path
    (both patch optimized_attention_override and can chain). The block-level
    ltx2/wan/minimax_h3 sage patches bypass that hook entirely, so combining them
    with sol_attn=True is a silent no-op for Sol-Attn -- this node raises instead
    of allowing that combination quietly.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sage_architecture": (
                    ["off", "auto", "generic", "ltx2", "wan", "minimax_h3"],
                    {"default": "auto", "tooltip": "'off' disables sage attention entirely. 'auto' detects arch, falling back to 'generic'."},
                ),
                "sol_attn": ("BOOLEAN", {"default": False, "tooltip": "Enable Sol-Attn sparse routing (requires ComfyUI-SolAttn_triton installed). Only compatible with sage_architecture in {off, generic, auto-that-resolves-to-generic}."}),
            },
            "optional": {
                "sage_attention_mode": (sageattn_modes, {"default": "auto"}),
                "allow_compile": ("BOOLEAN", {"default": False}),
                "triton_kernels": ("BOOLEAN", {"default": True}),
                "minimax_head_chunks": ("INT", {"default": 1, "min": 1, "max": 64}),
                "sol_tau": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Sol-Attn sparsity threshold. Higher = sparser/faster/lower fidelity."}),
                "sol_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "GRNodes/experimental"
    DESCRIPTION = (
        "Sage attention + Sol-Attn in one node with independent switches for each. "
        "Spectrum is NOT folded in here (its forecasting logic is genuinely different "
        "per model backend) -- chain the matching ComfyUI-Spectrum-* node after this "
        "one; it patches via model_function_wrapper so it composes fine either way."
    )
    EXPERIMENTAL = True

    def patch(self, model, sage_architecture, sol_attn, sage_attention_mode="auto",
              allow_compile=False, triton_kernels=True, minimax_head_chunks=1,
              sol_tau=1.2, sol_start_percent=0.2, sol_end_percent=0.9):

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        resolved_sage = sage_architecture
        if resolved_sage == "auto":
            resolved_sage = GRUnifiedSageAttentionPatch._detect_architecture(diffusion_model)

        if sol_attn and resolved_sage not in ("off", "generic"):
            raise RuntimeError(
                f"sol_attn=True with sage_architecture resolving to '{resolved_sage}': "
                "Sol-Attn only sees traffic through the generic optimized_attention_override "
                "hook. The block-level sage patch for this architecture bypasses that hook, "
                "so Sol-Attn would silently do nothing. Use sage_architecture='off' or "
                "'generic' if you want sol_attn=True, or turn sol_attn off."
            )

        # --- sage (generic or off) ---
        if resolved_sage == "generic":
            if sage_attention_mode != "disabled":
                new_attention = get_sage_func(sage_attention_mode, allow_compile=allow_compile)

                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)

                model_clone.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
        elif resolved_sage != "off":
            # ltx2 / wan / minimax_h3 block-level -- delegate to the single-purpose node
            sage_node = GRUnifiedSageAttentionPatch()
            (model_clone,) = sage_node.patch(
                model_clone, resolved_sage, sage_attention_mode=sage_attention_mode,
                allow_compile=allow_compile, triton_kernels=triton_kernels,
                minimax_head_chunks=minimax_head_chunks,
            )

        # --- sol-attn (chains onto whatever override is already installed) ---
        if sol_attn:
            if solattn_mod is None:
                raise RuntimeError(
                    "sol_attn=True but ComfyUI-SolAttn_triton is not installed. "
                    "Install it from https://github.com/kijai/ComfyUI-SolAttn_triton "
                    "or set sol_attn=False."
                )
            model_sampling = model_clone.get_model_object("model_sampling")
            sigma_start = float(model_sampling.percent_to_sigma(sol_start_percent))
            sigma_end = float(model_sampling.percent_to_sigma(sol_end_percent))
            previous = model_clone.model_options["transformer_options"].get("optimized_attention_override")
            if previous is not None:
                logging.info("GRUnifiedAccelerator: Sol-Attn chaining onto the sage override -- Sol-Attn gets first refusal, sage handles what Sol-Attn declines")
            model_clone.model_options["transformer_options"]["optimized_attention_override"] = solattn_mod.make_override(
                tau=sol_tau, sigma_start=sigma_start, sigma_end=sigma_end, previous=previous,
            )

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "GRUnifiedSageAttentionPatch": GRUnifiedSageAttentionPatch,
    "GRUnifiedAccelerator": GRUnifiedAccelerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GRUnifiedSageAttentionPatch": "GR Unified Sage Attention Patch",
    "GRUnifiedAccelerator": "GR Unified Accelerator (Sage + Sol-Attn)",
}
