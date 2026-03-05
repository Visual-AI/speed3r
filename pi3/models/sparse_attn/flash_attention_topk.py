import torch
import triton
import triton.language as tl
import time
from .topk_utils import *
import pdb 
import math

from .utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_with_topk(
    Q, K, V, Out,
    Lse,
    Topk_indices,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    stride_topk_i_b, stride_topk_i_h, stride_topk_i_m,
    nheads,
    seqlen_q, seqlen_k,
    seqlen_q_rounded,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_VAL: tl.constexpr,
):
    # pdb.set_trace()
    # --- Standard Initialization ---
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (tl.arange(0, BLOCK_N)[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (tl.arange(0, BLOCK_N)[:, None] * stride_vn + offs_d[None, :])
    
    # --- Accumulator Initialization ---
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    
    # Load q block
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

    # --- Main loop over K blocks ---
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)


    # variable for stream topk sorting
    x_nbits: tl.constexpr = 32
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    qk_topk = tl.zeros([BLOCK_M, K_VAL], dtype=tl.uint64)
    acc_qk = tl.zeros([BLOCK_M, K_VAL], dtype=tl.uint64)
    for start_n in tl.range(0, end_n, BLOCK_N):
        # Load k and compute qk
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        
        # Apply masks
        if not EVEN_N:
            qk += tl.where(offs_n[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

        # --- Online Softmax 
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_HEADDIM:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

        # online Top-K Computation ---
        
        # sort previous result to make it ascending for bitonic merge
        acc_qk = bitonic_merge(acc_qk, descending=False)
        
        # Convert qk to unsigned integer keys for stable sorting
        val_bits = qk.to(x_utype, bitcast=True)
        val_keys = fpval_to_key(val_bits).to(tl.uint64)

        # Create a matrix of indices for stable sorting
        indices_matrix = tl.broadcast_to(offs_n[None, :], qk.shape).to(tl.uint32)   
        stable_indices = (end_n - 1) - indices_matrix

        # Create composite keys for stable sorting with indices
        composite_keys = val_keys << 32 | stable_indices.to(tl.uint32)
        mask_n_topk = offs_n[None, :] < seqlen_k
        composite_keys = tl.where(mask_n_topk, composite_keys, 0)
        qk_topk = topk(composite_keys, k=K_VAL, descending=True)

        # bitonic merge the top-k results
        acc_qk = tl.maximum(acc_qk, qk_topk)

    # --- Finalization and Write-back ---
    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i, mask=offs_m < seqlen_q)
    
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    # --- Write back final Top-K results ---
    acc_qk = bitonic_merge(acc_qk, descending=True)
    topk_indices = ((end_n - 1) - (acc_qk & 0xFFFFFFFF)).to(tl.int32)
    offs_k = tl.arange(0, K_VAL)
    Topk_ind_ptr = Topk_indices + off_b * stride_topk_i_b + off_h * stride_topk_i_h + (offs_m[:, None] * stride_topk_i_m + offs_k[None, :])
    # tl.store(Topk_ind_ptr, topk_indices)
    mask_m = offs_m < seqlen_q

    tl.store(Topk_ind_ptr, topk_indices, mask=mask_m[:, None])



def forward_with_topk(q, k, v, topk_val, causal=False, softmax_scale=None):
    """ Wrapper for the forward pass of FlashAttention with Top-K functionality. """
    shape = q.shape
    batch_size, nheads, seqlen_q, headdim = shape
    _, _, seqlen_k, _ = k.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (headdim ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_HEADDIM = 64
    assert topk_val <= BLOCK_N, f"Top-K value ({topk_val}) cannot exceed BLOCK_N ({BLOCK_N})"
    assert headdim <= BLOCK_HEADDIM, "headdim must be less than or equal to BLOCK_HEADDIM"

    o = torch.empty_like(q)
    seqlen_q_rounded = triton.cdiv(seqlen_q, BLOCK_M) * BLOCK_M
    lse = torch.empty((batch_size, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    # topk_values = torch.empty((batch_size, nheads, seqlen_q, k_val), device=q.device, dtype=torch.float32)
    topk_indices = torch.full((batch_size, nheads, seqlen_q, topk_val), -1, device=q.device, dtype=torch.int32)

    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_size * nheads)
    # tmp = torch.empty((batch_size * nheads, seqlen_q), device=q.device, dtype=torch.float32)
    _fwd_kernel_with_topk[grid](
        q, k, v, o,
        lse,
        topk_indices,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2),
        nheads,
        seqlen_q, seqlen_k,
        seqlen_q_rounded,
        headdim,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        K_VAL=topk_val,
    )
    
    return o, lse, topk_indices, softmax_scale


@triton.jit
def backward_sum_o_do(
    o_ptr,  # O: [batch, nheads, seq_len, HEAD_DIM]
    do_ptr,  # dO: [batch, nheads, seq_len, HEAD_DIM]
    delta_ptr,  # D: [batch, seq_len, nheads]
    nheads,
    o_len,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_dob,
    stride_doh,
    stride_dol,
    stride_db,
    stride_dh,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_d = tl.arange(0, HEAD_DIM)


    offs_m = start_m * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    mask_m = offs_m < o_len

    off_o = o_ptr + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_ol + offs_d[None, :])
    off_do = do_ptr + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dol + offs_d[None, :])

    o  = tl.load(off_o, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(off_do, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=-1)
    tl.store(
        delta_ptr + off_b * stride_db + off_h * stride_dh + offs_m, 
        delta,
        mask=mask_m
    )


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: b x h x l x d
    k_ptr,  # K: b x h x l x d
    v_ptr,  # V: b x h x l x d
    lse_ptr,  # LSE: b x h x l
    delta_ptr,    # Delta: b x h x l
    do_ptr,
    dk_ptr,  # DK: b x h x l x d
    dv_ptr,  # DK: b x h x l x d
    # shape
    seq_len,
    nheads,
    # sm_scale
    sm_scale,
    # stride
    stride_qb, stride_qh, stride_ql, 
    stride_kb, stride_kh, stride_kl, 
    stride_vb, stride_vh, stride_vl, 
    stride_lb, stride_lh,
    stride_db, stride_dh,
    stride_dob, stride_doh, stride_dol,
    stride_dkb, stride_dkh, stride_dkl,
    stride_dvb, stride_dvh, stride_dvl,
    # META parameters
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale
    
    # Get key block ID and batch/head ID
    key_block_id = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    # For sparse attention, the key block position is simply key_block_id * BLOCK_SIZE_K
    key_seq_start = key_block_id * BLOCK_SIZE_K

    k_ptrs = tl.make_block_ptr(
        base=k_ptr + off_b * stride_kb + off_h * stride_kh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_kl, 1),
        offsets=(key_seq_start, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    v_ptrs = tl.make_block_ptr(
        base=v_ptr + off_b * stride_vb + off_h * stride_vh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_vl, 1), 
        offsets=(key_seq_start, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # Base pointers for query-related tensors
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    d_base = delta_ptr + off_b * stride_db + off_h * stride_dh
    lse_base = lse_ptr + off_b * stride_lb + off_h * stride_lh

    # Load K/V - add bounds checking
    k = tl.load(k_ptrs, boundary_check=(0, 1))
    v = tl.load(v_ptrs, boundary_check=(0, 1))

    # Initialize dk/dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

    # Iterate through all queries of current key block
    for q_seq_start in tl.range(0, seq_len, BLOCK_SIZE_Q):
        # Create block pointers for query data
        q_ptrs = tl.make_block_ptr(
            base=q_base,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_ql, 1),
            offsets=(q_seq_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
            order=(1, 0),
        )
        
        do_ptrs = tl.make_block_ptr(
            base=do_base,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_dol, 1),
            offsets=(q_seq_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
            order=(1, 0),
        )
        
        # Load query-related data with bounds checking
        q = tl.load(q_ptrs, boundary_check=(0, 1))
        do = tl.load(do_ptrs, boundary_check=(0, 1))
        
        # Load LSE and delta using offset calculations
        offs_q = q_seq_start + tl.arange(0, BLOCK_SIZE_Q)
        mask_q = offs_q < seq_len
        lse = tl.load(lse_base + offs_q, mask=mask_q, other=0.0)
        d = tl.load(d_base + offs_q, mask=mask_q, other=0.0)
        
        # Compute attention weights and gradients
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = qk_scale * p * (dp - d[:, None])
        
        # Type casting
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        
        # Accumulate gradients
        dk += tl.dot(tl.trans(ds), q)
        dv += tl.dot(tl.trans(p), do)

    # Store to correct positions with bounds checking
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + off_b * stride_dkb + off_h * stride_dkh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_dkl, 1),
        offsets=(key_seq_start, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + off_b * stride_dvb + off_h * stride_dvh,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_dvl, 1),
        offsets=(key_seq_start, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))



@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seq_len"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seq_len"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["HEAD_DIM"] == args["HEAD_DIM"],
    }
)
@triton.jit
def backward_dq(
    Q,  # Q: b x h x l x d
    K,  # K: b x h x l x d
    V,  # V: b x h x l x d
    lse_ptr,  # LSE: B x h x L
    d_ptr,  # Delta: B x h x L
    do_ptr, # dO: b x h x l x d
    dq_ptr, # dQ: b x h x l x d
    # shape
    seq_len,
    nheads,
    # sm_scale
    sm_scale,
    # stride
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_lb, stride_lh,
    stride_db, stride_dh,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    # META parameters
    HEAD_DIM: tl.constexpr,  # head dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads


    # # Bounds check for query block
    # if start_m * BLOCK_M >= seq_len:
    #     return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Query block pointers
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    do_ptrs = do_ptr + off_b * stride_dob + off_h * stride_doh + (offs_m[:, None] * stride_dom + offs_d[None, :])
    
    # Scalar pointers for LSE and delta
    d_ptrs = d_ptr + off_b * stride_db + off_h * stride_dh + offs_m
    lse_ptrs = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m
    
    # Load query block with proper masking
    mask_m = offs_m < seq_len
    mask_d = offs_d < HEAD_DIM
    
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load LSE and delta with masking
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    d = tl.load(d_ptrs, mask=mask_m, other=0.0)
    
    # Initialize dq with correct dimensions: (BLOCK_M, HEAD_DIM)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Base pointers for K and V
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    
    # Loop through topk key blocks

    for k_seq_start in tl.range(0, seq_len, BLOCK_N):
        # Create block pointers for K and V
        offs_n = k_seq_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        k_ptrs = k_base + (offs_n[:, None] * stride_kn + offs_d[None, :])
        v_ptrs = v_base + (offs_n[:, None] * stride_vn + offs_d[None, :])
        
        # Load K and V blocks with proper masking
        if EVEN_N & EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        
        # Apply causal mask if needed (optional)
        # qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        
        # Compute attention probabilities
        p = tl.exp(qk - lse[:, None])
        
        # Compute dp and ds
        dp = tl.dot(do, tl.trans(v))
        ds = qk_scale * p * (dp - d[:, None])
        
        # Cast to correct dtype
        ds = ds.to(q.dtype)
        
        # Accumulate dq gradient
        dq += tl.dot(ds, k)

    # Store dq with proper masking
    dq_ptrs = dq_ptr + off_b * stride_dqb + off_h * stride_dqh + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    
    if EVEN_M & EVEN_HEADDIM:
        tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty))
    else:
        tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


def _flash_attn_backward(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()

    batch, nheads, seq_len, HEAD_DIM = q.shape
    _, _, seqlen_k, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    # assert d in {16, 32, 64, 128}
    delta = torch.zeros([batch, nheads, seq_len], device=o.device, dtype=torch.float32)
    BLOCK_SIZE_O = 64
    BLOCK_SIZE_D = triton.next_power_of_2(HEAD_DIM)
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_SIZE_O, IS_HOPPER_GPU)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_O), batch * nheads)
    backward_sum_o_do[grid](
        o,
        do,
        delta,
        nheads,
        seq_len,
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1),
        HEAD_DIM,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # compute dk dv
    dk = torch.zeros(
        batch, nheads, seq_len, HEAD_DIM, device=k.device, dtype=k.dtype
    )
    dv = torch.zeros(
        batch, nheads, seq_len, HEAD_DIM, device=k.device, dtype=k.dtype
    )
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_D = triton.next_power_of_2(HEAD_DIM)
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_SIZE_Q, IS_HOPPER_GPU)

    grid = (triton.cdiv(seqlen_k, BLOCK_SIZE_K), batch * nheads)

    backward_dkdv[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dk,
        dv,
        seq_len,
        nheads,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        do.stride(0), do.stride(1), do.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        HEAD_DIM = HEAD_DIM,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # compute dq
    dq = torch.zeros_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_SIZE_K, IS_HOPPER_GPU)

    grid_dq = (triton.cdiv(seq_len, BLOCK_M), batch * nheads)

    backward_dq[grid_dq](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dq,
        seq_len,
        nheads,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        lse.stride(0), lse.stride(1),
        delta.stride(0), delta.stride(1),
        do.stride(0), do.stride(1), do.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return dq, dk, dv


class FlashAttnTopkFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, topk_val, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse, topk_indices, ctx.softmax_scale = forward_with_topk(
            q, k, v, topk_val, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = softmax_scale
        return o, topk_indices

    @staticmethod
    def backward(ctx, do, dtopk_indices):
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        dq, dk, dv = _flash_attn_backward(
            o,
            do,
            lse,
            q,
            k,
            v,
            sm_scale
        )
        return dq, dk, dv, None, None, None


flash_attn_topk_func = FlashAttnTopkFunc.apply
