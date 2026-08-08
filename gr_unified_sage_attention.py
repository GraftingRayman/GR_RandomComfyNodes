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


import math

# ---------------------------------------------------------------------------
# Spectrum (SDXL variant) -- vendored verbatim from ruwwww/ComfyUI-Spectrum-sdxl
# (src/spectrum_node.py + src/forecaster.py, read in full). This is the SIMPLE
# variant: patches set_model_unet_function_wrapper, forecasts raw UNet output
# via blended Chebyshev+local-Taylor regression. It is architecture-agnostic
# at the mechanism level (operates on whatever tensor the wrapper receives)
# but was calibrated/tested against SDXL.
#
# NOT included: the "Proper" / Flux / Wan / MiniMax-H3 Spectrum variants.
# Their runtime (solver-step scheduling, batch-label reordering across cond/
# uncond branches, forecast-invalidation on a dozen different invariants) is
# real production complexity across several more files I have not fully read
# -- vendoring it half-verified risks silently corrupted latents, which is
# worse than leaving those as their own separate nodes in your graph.
# ---------------------------------------------------------------------------

_SPECTRUM_DTYPE = torch.bfloat16


def _spectrum_flatten(x):
    return (x.reshape(1, -1) if x.ndim == 1 else x.reshape(1, -1)), x.shape


def _spectrum_unflatten(x_flat, shape):
    return x_flat.reshape(shape)


class _SpectrumBaseForecaster(torch.nn.Module):
    def __init__(self, M=3, K=10, lam=1e-3, device=None, feature_shape=None, t_max=50.0):
        super().__init__()
        assert K >= M + 2, "K should exceed basis size for stability"
        self.M, self.K, self.lam, self.t_max_val = M, K, lam, t_max
        self.register_buffer("t_buf", torch.empty(0))
        self._H_buf = None
        self._shape = None
        self._coef = None
        self._XtX_fac = None
        self._tau_cache = None
        self._X_cache = None
        self._last_delta_norm = None
        self.device_ref = device
        self.feature_shape = feature_shape

    def _taus(self, t):
        assert self.t_buf.numel() >= 1
        t_min = torch.zeros(1, device=t.device, dtype=t.dtype)
        t_max = torch.ones(1, device=t.device, dtype=t.dtype) * self.t_max_val
        if torch.isclose(t_max, t_min):
            return torch.zeros_like(t)
        mid = 0.5 * (t_min + t_max)
        rng = (t_max - t_min)
        return (t - mid) * 2.0 / rng

    def _build_design(self, taus):
        raise NotImplementedError

    def update(self, t, h):
        device = self.device_ref or h.device
        t = torch.as_tensor(t, dtype=_SPECTRUM_DTYPE, device=device)
        h_flat, shape = _spectrum_flatten(h)
        h_flat = h_flat.to(device)
        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"
        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            delta = (h_flat - self._H_buf[-1])
            self._last_delta_norm = delta.norm(p=2)
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self._H_buf = self._H_buf[-self.K:]
        self._coef = self._XtX_fac = self._tau_cache = self._X_cache = None

    def ready(self):
        return self.t_buf.numel() >= min(self.K, self.M + 2)

    def _fit_if_needed(self):
        if self._coef is not None:
            return
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)
        H = self._H_buf.to(torch.float32)
        _, P = X.shape
        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.transpose(0, 1)
        XtX = Xt @ X + lamI
        try:
            L = torch.linalg.cholesky(XtX)
        except Exception:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))
        XtH = Xt @ H
        self._coef = torch.cholesky_solve(XtH, L).to(_SPECTRUM_DTYPE)
        self._XtX_fac, self._tau_cache, self._X_cache = L, taus, X.to(_SPECTRUM_DTYPE)

    @torch.no_grad()
    def predict(self, t_star):
        assert self._shape is not None
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=_SPECTRUM_DTYPE, device=device)
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])
        return _spectrum_unflatten(x_star @ self._coef, self._shape)


class _SpectrumChebyshevForecaster(_SpectrumBaseForecaster):
    def _build_design(self, taus):
        taus = taus.reshape(-1, 1)
        T0 = torch.ones((taus.shape[0], 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0
        cols = [T0, taus]
        for _ in range(2, self.M + 1):
            cols.append(2 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[: self.M + 1], dim=1)


class _Spectrum(torch.nn.Module):
    def __init__(self, cheb_like, taylor_order=1, w=None):
        super().__init__()
        self.cheb = cheb_like
        self.taylor_order = taylor_order
        self.w = w

    @torch.no_grad()
    def _local_taylor_discrete(self, t_star):
        H, t = self.cheb._H_buf, self.cheb.t_buf
        h_i, t_i = H[-1], t[-1]
        if t.numel() < 2:
            return _spectrum_unflatten(h_i.clone().reshape(1, -1), self.cheb._shape)
        h_im1, t_im1 = H[-2], t[-2]
        dh1 = h_i - h_im1
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)
        out = h_i + k * dh1
        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = H[-3]
            out = out + 0.5 * k * (k - 1.0) * (h_i - 2 * h_im1 + h_im2)
        return _spectrum_unflatten(out.reshape(1, -1), self.cheb._shape)

    @torch.no_grad()
    def predict(self, t_star):
        device = self.cheb.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=_SPECTRUM_DTYPE, device=device)
        h_taylor = self._local_taylor_discrete(t_star)
        if not self.cheb.ready():
            return h_taylor
        h_cheb = self.cheb.predict(t_star)
        return (1 - self.w) * h_taylor + self.w * h_cheb

    def update(self, t, h):
        return self.cheb.update(t, h)

    def ready(self):
        return self.cheb.ready()


def _spectrum_sdxl_wrap_model(model, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step, steps):
    """Faithful port of SpectrumSDXL.patch() -- installs a unet_function_wrapper."""
    state = {
        "forecasters": None, "cnt": 0, "num_cached": [0],
        "curr_ws": float(window_size), "last_t": -1, "total_runs": 0,
        "estimated_total_steps": steps,
    }
    forecast_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def spectrum_unet_wrapper(model_function, kwargs):
        x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
        batch_size = x.shape[0]

        if not state.get("_shape_logged"):
            state["_shape_logged"] = True
            if x.ndim == 5:
                per_row_mb = x[0].numel() * x.element_size() / (1024 * 1024)
                est_mb = per_row_mb * min(100, 1) * batch_size  # K capped effectively by ready()/M+2, rough estimate
                logging.warning(
                    f"GRSpectrumApply: 5D (video-shaped) latent detected {tuple(x.shape)}. "
                    "This node's row-identity protection is a lightweight shape-based "
                    "fingerprint, NOT the full transactional branch-tracking the dedicated "
                    "ComfyUI-Spectrum-WAN-Proper / -MiniMax-H3 repos implement. Verify output "
                    "quality against spectrum-disabled before trusting this for real work. "
                    f"Approx per-step-per-row history footprint: {per_row_mb:.1f} MB."
                )

        t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) and timestep.numel() > 0 else float(timestep)

        if t_scalar > state["last_t"]:
            state["forecasters"] = None
            state["cnt"] = 0
            state["num_cached"] = [0] * batch_size
            state["curr_ws"] = float(window_size)
            state["total_runs"] += 1
            state["fingerprints"] = [None] * batch_size
        state["last_t"] = t_scalar

        if state["forecasters"] is None:
            state["forecasters"] = [
                _Spectrum(cheb_like=_SpectrumChebyshevForecaster(M=m, K=100, lam=lam, t_max=float(steps)), w=w)
                for _ in range(batch_size)
            ]
        if len(state["num_cached"]) != batch_size:
            state["num_cached"] = [0] * batch_size
        if len(state.get("fingerprints", [])) != batch_size:
            state["fingerprints"] = [None] * batch_size

        # Cheap batch-identity guard: this SDXL-style implementation assumes batch
        # index i means the same conditioning branch on every step. That's a safe
        # assumption for a single-frame image UNet call, but is exactly the thing
        # the real Wan/MiniMax-H3 Spectrum repos build whole transactional
        # branch-label systems to guarantee, because samplers CAN reorder or
        # resize cond/uncond rows between steps on video models. We don't have
        # that machinery here. What we do instead: fingerprint each batch row by
        # the shapes of its conditioning tensors (cheap -- no tensor hashing) and
        # if index i's fingerprint changes mid-run, force a REAL step for that
        # index instead of forecasting against a history that may belong to a
        # different conditioning branch now. This catches the coarse "rows got
        # reordered/resized" failure mode; it does NOT catch same-shape rows
        # silently swapping identity, which the real branch-label systems do
        # catch. Treat this as reduced protection, not equivalent protection.
        def _row_fingerprint(idx):
            parts = []
            for key in sorted(c.keys()):
                v = c[key]
                if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                    parts.append((key, tuple(v[idx].shape)))
                elif isinstance(v, torch.Tensor):
                    parts.append((key, tuple(v.shape)))
            return tuple(parts)

        fingerprint_mismatch = [False] * batch_size
        for i in range(batch_size):
            fp = _row_fingerprint(i)
            if state["fingerprints"][i] is not None and state["fingerprints"][i] != fp:
                fingerprint_mismatch[i] = True
                logging.warning(
                    f"GRSpectrumApply: conditioning shape at batch index {i} changed "
                    f"mid-run (t={t_scalar}) -- forcing a real step for this index "
                    "instead of forecasting, since cached history may belong to a "
                    "different conditioning branch now."
                )
            state["fingerprints"][i] = fp

        do_actual = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            if fingerprint_mismatch[i]:
                continue  # stays True -- forced real step
            is_micro_final = False
            if stop_caching_step == -1:
                if state["cnt"] >= int(state["estimated_total_steps"] * 0.8):
                    is_micro_final = True
            elif stop_caching_step > 0 and state["cnt"] >= stop_caching_step:
                is_micro_final = True
            if state["cnt"] >= warmup_steps and not is_micro_final:
                if state["forecasters"][i].ready():
                    do_actual[i] = (state["num_cached"][i] + 1) % math.floor(state["curr_ws"]) == 0
                else:
                    do_actual[i] = True

        real_mask, forecast_mask = do_actual, ~do_actual
        out = torch.empty_like(x)

        if real_mask.any():
            x_real = x[real_mask]
            timestep_real = timestep[real_mask.to(timestep.device)] if isinstance(timestep, torch.Tensor) and timestep.shape[0] == batch_size else timestep
            c_real = {k: v[real_mask.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v for k, v in c.items()}
            with torch.cuda.stream(torch.cuda.default_stream()):
                raw_real = model_function(x_real, timestep_real, **c_real)
            out[real_mask] = raw_real
            real_indices = real_mask.nonzero().squeeze()
            real_indices = [real_indices.item()] if real_indices.dim() == 0 else real_indices.tolist()
            for i, idx in enumerate(real_indices):
                state["forecasters"][idx].update(state["cnt"], raw_real[i])
                state["num_cached"][idx] = 0

        if forecast_mask.any():
            forecast_indices = forecast_mask.nonzero().squeeze()
            forecast_indices = [forecast_indices.item()] if forecast_indices.dim() == 0 else forecast_indices.tolist()
            out_forecast = torch.empty((len(forecast_indices), *x.shape[1:]), device=x.device, dtype=x.dtype)

            def _do_forecast():
                for j, i in enumerate(forecast_indices):
                    out_forecast[j] = state["forecasters"][i].predict(state["cnt"])
                out[forecast_mask] = out_forecast
                for i in forecast_indices:
                    state["num_cached"][i] += 1

            if forecast_stream:
                with torch.cuda.stream(forecast_stream):
                    _do_forecast()
                torch.cuda.current_stream().wait_stream(forecast_stream)
            else:
                _do_forecast()

        if state["cnt"] >= warmup_steps:
            state["curr_ws"] += flex_window
        state["cnt"] += 1
        return out

    new_model = model.clone()
    new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)
    return new_model


class GRSpectrumApply:
    """
    Generic Spectrum sampling-step forecaster (Chebyshev + local-Taylor blend),
    vendored from ruwwww/ComfyUI-Spectrum-sdxl. Skips computing the real network
    output on some fraction of steps and forecasts it instead, once enough
    history is built up.

    SCOPE: verified-reasonable for single-frame/image models (SDXL, Flux,
    Z-Image). Will run on video-shaped (5D) latents too since the underlying
    math is shape-agnostic, but the row-identity protection here is a cheap
    shape-based fingerprint, NOT the full transactional cond/uncond
    branch-tracking that the dedicated ComfyUI-Spectrum-WAN-Proper and
    ComfyUI-Spectrum-MiniMax-H3 repos implement for exactly this reason.
    On video, treat this as experimental: compare fixed-seed output against
    spectrum_enabled=False before trusting it, and prefer the dedicated repos
    for anything you're not willing to manually verify.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "tooltip": "Your sampler's total step count -- must match, since the forecaster's time axis is calibrated to it."}),
            },
            "optional": {
                "w": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blend weight toward the Chebyshev global fit vs local-Taylor extrapolation."}),
                "m": ("INT", {"default": 3, "min": 0, "max": 8, "tooltip": "Chebyshev polynomial order."}),
                "lam": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 20.0, "step": 0.5, "tooltip": "Forecast this many steps before doing another real one."}),
                "flex_window": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "tooltip": "Grow (or shrink) the window each step after warmup."}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100, "tooltip": "Run this many real steps before forecasting starts."}),
                "stop_caching_step": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Force real steps from here on. -1 = auto (last 20% of steps)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "GRNodes/experimental"
    DESCRIPTION = (
        "Sampling-step forecaster (Spectrum, SDXL-derived). Verified-reasonable for "
        "image models; runs on video models too but with reduced row-identity "
        "protection vs the dedicated Wan/MiniMax-H3 Spectrum repos -- verify output "
        "quality before trusting it there."
    )
    EXPERIMENTAL = True

    def patch(self, model, steps, w=0.5, m=3, lam=1e-3, window_size=2.0,
              flex_window=0.0, warmup_steps=3, stop_caching_step=-1):
        return (_spectrum_sdxl_wrap_model(
            model, w=w, m=m, lam=lam, window_size=window_size,
            flex_window=flex_window, warmup_steps=warmup_steps,
            stop_caching_step=stop_caching_step, steps=steps,
        ),)


try:
    # Vendored against the REAL kijai/ComfyUI-SolAttn_triton __init__.py (fetched
    # and read in full -- not the earlier guessed API). We still import the
    # installed package rather than vendor it: its Triton kernels (_tri_fwd,
    # _int8_fwd) and the Morton reordering hooks (_morton, _morton_h3) are real
    # proprietary compute we haven't seen and shouldn't fake.
    for _pkg in ("custom_nodes.ComfyUI-SolAttn_triton", "ComfyUI-SolAttn_triton"):
        try:
            solattn_mod = importlib.import_module(_pkg)
            break
        except ImportError:
            solattn_mod = None
except Exception:
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
                "sol_tau": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Threshold beta. Higher is sparser: 1.0 ~ 16% of blocks kept exact, 1.5 ~ 7%, 2.0 ~ 2.7%."}),
                "sol_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Run dense before this point. The paper uses 0.2."}),
                "sol_end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_min_tokens": ("INT", {"default": 4096, "min": 0, "max": 1 << 20, "step": 512, "tooltip": "Sequences shorter than this stay dense."}),
                "sol_int8_qk": ("BOOLEAN", {"default": False, "tooltip": "INT8 QK in the exact branch. Free in quality at tau<=1.5, a net loss at tau>=2.0."}),
                "sol_sink_conditioning": (["exact_kv", "exact_kv_and_rows", "off"], {"default": "exact_kv", "tooltip": "MiniMax-H3 only. exact_kv: packed text/audio/reference rows exact (~3% cost). exact_kv_and_rows: also runs those query rows dense (~20% cost, exact audio). No effect on other models."}),
                "sol_morton": ("BOOLEAN", {"default": True, "tooltip": "Reorder video tokens into Morton (Z-order) so each 64-token block is a compact 3D neighbourhood -- makes routing far more accurate at a given density. Wan and MiniMax-H3 only; skipped elsewhere."}),
                "sol_morton_curve": (["3d", "2d_frame"], {"default": "3d", "tooltip": "3d interleaves t/h/w equally. 2d_frame Z-orders within each frame -- try this if 3d degrades at some frame counts (e.g. MiniMax-H3's non-uniform frame spacing)."}),
                "sol_verbose": ("BOOLEAN", {"default": False, "tooltip": "Log Sol-Attn's per-shape sparse/dense dispatch decisions."}),
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
              sol_tau=1.2, sol_start_percent=0.2, sol_end_percent=0.9,
              sol_min_tokens=4096, sol_int8_qk=False, sol_sink_conditioning="exact_kv",
              sol_morton=True, sol_morton_curve="3d", sol_verbose=False):

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
                    "sol_attn=True but ComfyUI-SolAttn_triton is not installed (or its "
                    "package folder name doesn't match what this node tries to import -- "
                    "check the custom_nodes folder name matches 'ComfyUI-SolAttn_triton'). "
                    "Install it from https://github.com/kijai/ComfyUI-SolAttn_triton "
                    "or set sol_attn=False."
                )
            if solattn_mod._sol_attn_kernel is None:
                raise RuntimeError(f"Sol-Attn kernel unavailable: {solattn_mod._IMPORT_ERROR}")
            if sol_int8_qk and solattn_mod._sol_attn_int8_kernel is None:
                raise RuntimeError(f"Sol-Attn INT8 kernel unavailable: {solattn_mod._INT8_IMPORT_ERROR}")

            diffusion_model = model_clone.get_model_object("diffusion_model")
            is_h3 = hasattr(diffusion_model, "rope_freqs") and hasattr(diffusion_model, "_forward")
            is_wan = hasattr(diffusion_model, "rope_encode") and hasattr(diffusion_model, "blocks")

            # H3 needs the segment-layout hooks installed for the conditioning sink even
            # when Morton reordering itself is off -- mirrors kijai's execute() exactly.
            reorder = False
            if is_h3 and (sol_morton or sol_sink_conditioning != "off"):
                from importlib import import_module
                morton_h3 = import_module(f"{solattn_mod.__name__}._morton_h3")
                morton_h3.install_h3_hooks(diffusion_model)
                reorder = sol_morton
            elif is_wan and sol_morton:
                from importlib import import_module
                morton_wan = import_module(f"{solattn_mod.__name__}._morton")
                morton_wan.install_wan_morton(diffusion_model)
                reorder = True
            elif sol_morton:
                logging.warning(
                    f"GRUnifiedAccelerator: Morton reordering skipped -- "
                    f"{type(diffusion_model).__name__} is neither Wan-style nor MiniMax-H3. "
                    "Sol-Attn itself still applies."
                )

            model_sampling = model_clone.get_model_object("model_sampling")
            sigma_start = float(model_sampling.percent_to_sigma(sol_start_percent))
            sigma_end = float(model_sampling.percent_to_sigma(sol_end_percent))
            previous = model_clone.model_options["transformer_options"].get("optimized_attention_override")
            if previous is not None:
                logging.info("GRUnifiedAccelerator: Sol-Attn chaining onto the sage override -- Sol-Attn gets first refusal, sage handles what Sol-Attn declines")

            model_clone.model_options["transformer_options"]["optimized_attention_override"] = solattn_mod.make_override(
                tau=sol_tau, min_tokens=sol_min_tokens,
                sigma_start=sigma_start, sigma_end=sigma_end, verbose=sol_verbose,
                int8_qk=sol_int8_qk, sink_conditioning=sol_sink_conditioning, previous=previous,
            )
            if reorder:
                model_clone.model_options["transformer_options"]["sol_morton"] = True
                model_clone.model_options["transformer_options"]["sol_morton_curve"] = sol_morton_curve

            solattn_mod.reset_sol_attn_stats()

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "GRUnifiedSageAttentionPatch": GRUnifiedSageAttentionPatch,
    "GRUnifiedAccelerator": GRUnifiedAccelerator,
    "GRSpectrumApply": GRSpectrumApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GRUnifiedSageAttentionPatch": "GR Unified Sage Attention Patch",
    "GRUnifiedAccelerator": "GR Unified Accelerator (Sage + Sol-Attn)",
    "GRSpectrumApply": "GR Spectrum Apply (Generic, incl. video)",
}