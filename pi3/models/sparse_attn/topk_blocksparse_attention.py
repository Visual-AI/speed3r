# Copyright 2025 Xunhao Lai & Jianqiao Lu.
# Modifications Copyright 2026 Weining Ren.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from .utils import get_num_warps_stages, is_hopper_gpu
import pdb

IS_HOPPER_GPU = is_hopper_gpu()

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seq_len"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seq_len"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["HEAD_DIM"] == args["HEAD_DIM"],
    }
)
@triton.jit
def block_sparse_forward_kernel(
    Q,  # Q: b x h x l x d
    K,  # K: b x h x l x d
    V,  # V: b x h x l x d
    T,  # topk_idx: b x h x (l / block_m) x k
    O,  # O: b x h x l x d
    lse_ptr,  # LSE: B x h x L
    # sm_scale
    sm_scale,
    seq_len,
    # stride
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_tb, stride_th, stride_tm,
    stride_ob, stride_oh, stride_ol,
    stride_lb, stride_lh,
    # META parameters
    BLOCK_M: tl.constexpr,  # q block size
    BLOCK_N: tl.constexpr,  # k block size
    TOPK: tl.constexpr,  # topk number
    HEAD_DIM: tl.constexpr,  # head dimension
    nheads: tl.constexpr,  # number of heads
    block_size: tl.constexpr,  # block size
    # num_warps: tl.constexpr,  # number of warps
    # num_stages: tl.constexpr,  # number of stages
    EVEN_M: tl.constexpr,  # whether BLOCK_M is even
    EVEN_N: tl.constexpr,  # whether BLOCK_N is even
    EVEN_HEADDIM: tl.constexpr,  # whether HEAD_DIM is even
):
    # --- Standard Initialization ---
    # pdb.set_trace()
    # qk_scale = sm_scale * 1.44269504
    qk_scale = sm_scale
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seq_len

    
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    t_ptrs = T + off_b * stride_tb + off_h * stride_th + start_m * stride_tm

    # --- Accumulator Initialization ---
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load q block
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # main loop over topk
    for block_idx in tl.range(TOPK):
        # init topk idx pointer and get value
        topk_idx = tl.load(t_ptrs + block_idx) * block_size
        if topk_idx >= 0 and topk_idx < seq_len:  # Add upper bound check
            k_mask = (topk_idx + tl.arange(0, BLOCK_N)) < seq_len
            k = tl.load(k_ptrs + topk_idx * stride_kn, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptrs + topk_idx * stride_vn, mask=k_mask[:, None], other=0.0)
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k)) * qk_scale

            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]

            v = tl.load(v_ptrs + topk_idx * stride_vn, mask=k_mask[:, None], other=0.0)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            # update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)


    # final scale
    acc_o = acc_o * tl.exp(m_i - lse_i)[:, None]
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_ol + offs_d[None, :])
    tl.store(o_ptrs, acc_o.to(o_ptrs.dtype.element_ty), mask=mask_m[:, None])

    # save lse
    lse_ptr = lse_ptr + off_b * stride_lb + off_h * stride_lh + offs_m
    tl.store(lse_ptr, lse_i.to(lse_ptr.dtype.element_ty), mask=mask_m)


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
def save_topk_idx_kernel_fixedlen(
    p_ptr,              # Source: pad_topk_q_idx [b, h, seq_len * topk]
    t_ptr,              # Destination: topk_q_idx [b, h, max_connections]
    c_ptr,              # Offsets: cu_topk_q_count [b, h, num_blocks + 1]
    # Strides
    stride_pb, stride_ph, stride_pn,
    stride_tb, stride_th, stride_tn,
    stride_cb, stride_ch, stride_ck,
    # boundary check sie
    p_size: tl.constexpr,  # pad_topk_q_idx last dim
    t_size: tl.constexpr,  # topk_q_idx last dim
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    p_ptr_base = p_ptr + pid_b * stride_pb + pid_h * stride_ph
    t_ptr_base = t_ptr + pid_b * stride_tb + pid_h * stride_th
    c_ptr_base = c_ptr + pid_b * stride_cb + pid_h * stride_ch

    c_start = tl.load(c_ptr_base + pid_k * stride_ck)
    c_end = tl.load(c_ptr_base + (pid_k + 1) * stride_ck)
    c_len = c_end - c_start

    if c_len <= 0:
        return

    for i in tl.range(0, c_len, BLOCK_SIZE_M):
        off = i + tl.arange(0, BLOCK_SIZE_M)
        mask = off < c_len
        src_idx = c_start + off
        dst_idx = c_start + off
        
        # mask
        src_mask = mask & (src_idx < p_size)
        dst_mask = mask & (dst_idx < t_size)
        final_mask = src_mask & dst_mask
        
        p_vals = tl.load(p_ptr_base + src_idx * stride_pn, mask=final_mask, other=0)
        tl.store(t_ptr_base + dst_idx * stride_tn, p_vals, mask=final_mask)


# @triton.jit
# def count_kernel_fixedlen(
#     x_ptr,      # input: topk_idx [batch, head, seq_len, topk]
#     y_ptr,      # output: active_query_count [batch, head, num_blocks]
#     SEQ_LEN: tl.constexpr,
#     TOPK: tl.constexpr,
#     NUM_BLOCKS: tl.constexpr,
#     stride_xb, stride_xh, stride_xn,
#     stride_yb, stride_yh, stride_yk,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     BLOCK_SIZE_R: tl.constexpr,
# ):
#     pid_b = tl.program_id(0)  # batch_size dimension
#     pid_h = tl.program_id(1)  # num_kv_heads dimension


#     x_ptr_base = x_ptr + pid_b * stride_xb + pid_h * stride_xh
#     y_ptr_base = y_ptr + pid_b * stride_yb + pid_h * stride_yh

#     y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
    

#     for i in range(0, SEQ_LEN, BLOCK_SIZE_N):
#         off_n = i + tl.arange(0, BLOCK_SIZE_N)
#         off_k = tl.arange(0, BLOCK_SIZE_K)
#         x_ptrs = x_ptr_base + off_n[:, None] * stride_xn + off_k[None, :]
        
#         mask = (off_n[:, None] < SEQ_LEN) & (off_k[None, :] < TOPK)
#         x = tl.load(x_ptrs, mask=mask, other=-1) 
#         x = tl.where((x >= 0) & (x < NUM_BLOCKS), x, BLOCK_SIZE_R - 1)
#         x = tl.ravel(x)
#         y += tl.histogram(x, BLOCK_SIZE_R)

#     off_k_out = tl.arange(0, BLOCK_SIZE_R)
#     y_ptrs = y_ptr_base + off_k_out * stride_yk
#     tl.store(y_ptrs, y, mask=off_k_out < NUM_BLOCKS)



# def count_query_fixedlen(
#     topk_idx: torch.Tensor, # [batch, num_heads, seq_len, topk]
#     block_size: int
# ):
#     batch_size, num_heads, seq_len, topk = topk_idx.shape
    
#     max_block_id = topk_idx.max().item()
#     num_blocks_per_seq = max_block_id + 1  # 键块ID从0开始，所以+1
    
#     active_query_count = torch.zeros(
#         batch_size, num_heads, num_blocks_per_seq, 
#         dtype=torch.int32, device=topk_idx.device
#     )
    
#     BLOCK_SIZE_K = triton.next_power_of_2(topk)
#     BLOCK_SIZE_N = triton.next_power_of_2(4096 // BLOCK_SIZE_K)
#     BLOCK_SIZE_R = triton.next_power_of_2(num_blocks_per_seq)

#     grid = (batch_size, num_heads)

#     count_kernel_fixedlen[grid](
#         topk_idx,
#         active_query_count,
#         seq_len,
#         topk,
#         num_blocks_per_seq,
#         topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
#         active_query_count.stride(0), active_query_count.stride(1), active_query_count.stride(2),
#         BLOCK_SIZE_N=BLOCK_SIZE_N,
#         BLOCK_SIZE_K=BLOCK_SIZE_K,
#         BLOCK_SIZE_R=BLOCK_SIZE_R,
#         num_warps=4,
#         num_stages=3,
#     )
#     return active_query_count



# def reorder_topk_idx_fixedlen(
#     topk_idx: torch.Tensor,         # Shape: [batch, num_heads, seq_len, topk]
#     active_query_count: torch.Tensor, # Shape: [batch, num_heads, num_blocks]
#     block_size: int
# ) -> torch.Tensor:
#     """
#     Reorders topk_idx for the backward pass in a fixed-length (batched) setting.
#     It transforms the query-centric topk_idx into a key-centric index.

#     Args:
#         topk_idx: Tensor containing key block indices for each query.
#         active_query_count: Tensor from count_query_fixedlen, where [b, h, k] 
#                             is the number of queries in batch b, head h that 
#                             attend to key block k.
#         block_size: The size of the key/value blocks.

#     Returns:
#         A compact tensor `topk_q_idx` where query indices are grouped by the 
#         key block they attend to.
#     """
#     batch_size, num_kv_heads, seq_len, topk = topk_idx.shape
#     num_blocks = active_query_count.shape[-1]

#     # === Step 1: The `argsort` Trick ===
#     flat_topk_idx = topk_idx.reshape(batch_size, num_kv_heads, -1)
#     device = topk_idx.device
#     query_indices = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, topk).reshape(-1)
#     query_indices = query_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_kv_heads, -1)
#     sorted_indices = torch.argsort(flat_topk_idx, dim=-1, stable=True)
#     pad_topk_q_idx = torch.gather(query_indices, dim=-1, index=sorted_indices).to(torch.int32)

#     # === Step 2: Compact the Result ===

#     # (a) Calculate cumulative counts. This tells the kernel WHERE to write the data.
#     # cu_topk_q_count[b, h, k] will be the starting index for queries related to block k.
#     cu_topk_q_count = torch.nn.functional.pad(
#         active_query_count.cumsum(dim=-1), (1, 0)
#     ).to(torch.int32)
    
#     # Create the final output tensor. Its last dimension must be large enough
#     # to hold all connections for the "busiest" sample in the batch.
#     max_connections_per_sample = cu_topk_q_count[..., -1].max().item()
#     topk_q_idx = torch.full(
#         (batch_size, num_kv_heads, max_connections_per_sample),
#         fill_value=-1,  # Use -1 for padding
#         device=topk_idx.device,
#         dtype=torch.int32,
#     )


#     # save data
#     grid = (batch_size, num_kv_heads, num_blocks)    
#     BLOCK_SIZE_M = 256 

#     save_topk_idx_kernel_fixedlen[grid](
#         pad_topk_q_idx,         # Source (dense, sorted)
#         topk_q_idx,             # Destination (compact)
#         cu_topk_q_count,        # Offsets for reading and writing
#         # Strides
#         pad_topk_q_idx.stride(0), pad_topk_q_idx.stride(1), pad_topk_q_idx.stride(2),
#         topk_q_idx.stride(0), topk_q_idx.stride(1), topk_q_idx.stride(2),
#         cu_topk_q_count.stride(0), cu_topk_q_count.stride(1), cu_topk_q_count.stride(2),
#         # boundary check param
#         p_size=pad_topk_q_idx.shape[-1],
#         t_size=topk_q_idx.shape[-1],
#         BLOCK_SIZE_M=BLOCK_SIZE_M
#     )

#     return topk_q_idx

# =========================================================================
# === CORRECTED KERNEL AND LAUNCHER =======================================
# =========================================================================

@triton.jit
def count_kernel_fixedlen(
    x_ptr,      # input: topk_idx [batch, head, num_q_blocks, topk]
    y_ptr,      # output: active_query_count [batch, head, num_k_blocks]
    NUM_Q_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    stride_xb, stride_xh, stride_xm,
    stride_yb, stride_yh, stride_yk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    """
    Correctly counts active query blocks for each key block.
    Iterates over the query-block dimension, not the sequence dimension.
    """
    pid_b = tl.program_id(0)  # batch_size dimension
    pid_h = tl.program_id(1)  # num_heads dimension

    x_ptr_base = x_ptr + pid_b * stride_xb + pid_h * stride_xh
    y_ptr_base = y_ptr + pid_b * stride_yb + pid_h * stride_yh

    # Use a power-of-2 accumulator for the histogram for efficiency.
    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)

    # Loop over the query blocks dimension (M)
    for i in tl.range(0, NUM_Q_BLOCKS, BLOCK_SIZE_M):
        off_m = i + tl.arange(0, BLOCK_SIZE_M)
        off_k = tl.arange(0, BLOCK_SIZE_K)
        
        # Create pointers for a tile of the topk_idx tensor
        x_ptrs = x_ptr_base + off_m[:, None] * stride_xm + off_k[None, :]
        
        # Boundary check mask for the tile
        mask = (off_m[:, None] < NUM_Q_BLOCKS) & (off_k[None, :] < TOPK)
        
        # Load the key block indices, with padding set to -1
        x = tl.load(x_ptrs, mask=mask, other=-1)
        
        # Clamp indices to be valid for the histogram.
        # Any invalid index (padding < 0 or outlier >= NUM_K_BLOCKS) is
        # mapped to a garbage bin at the end of the accumulator.
        x = tl.where((x >= 0) & (x < NUM_K_BLOCKS), x, BLOCK_SIZE_R - 1)
        
        # Flatten the tile and update the histogram
        x = tl.ravel(x)
        y += tl.histogram(x, BLOCK_SIZE_R)

    # Store the results back to global memory
    off_k_out = tl.arange(0, BLOCK_SIZE_R)
    y_ptrs = y_ptr_base + off_k_out * stride_yk
    tl.store(y_ptrs, y, mask=off_k_out < NUM_K_BLOCKS)


def count_query_fixedlen(
    topk_idx: torch.Tensor, # [batch, num_heads, num_q_blocks, topk]
    block_size: int
):
    """
    Host-side launcher for the corrected count_kernel_fixedlen.
    """
    batch_size, num_heads, num_q_blocks, topk = topk_idx.shape
    
    # Determine the number of key blocks. This can be larger than num_q_blocks.
    # Add 1 because block IDs are 0-indexed.
    max_block_id = topk_idx.max().item()
    num_k_blocks = max_block_id + 1
    
    active_query_count = torch.zeros(
        batch_size, num_heads, num_k_blocks,
        dtype=torch.int32, device=topk_idx.device
    )
    
    # Kernel configuration
    BLOCK_SIZE_K = triton.next_power_of_2(topk)
    # This block size is for iterating over query blocks inside the kernel
    BLOCK_SIZE_M = 64 
    # Round up the number of key blocks to the next power of 2 for the histogram accumulator
    BLOCK_SIZE_R = triton.next_power_of_2(num_k_blocks)

    grid = (batch_size, num_heads)

    count_kernel_fixedlen[grid](
        topk_idx,
        active_query_count,
        # Pass constants to the kernel
        num_q_blocks,
        topk,
        num_k_blocks,
        # Strides
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2), # stride_xm is stride for the 3rd dim
        active_query_count.stride(0), active_query_count.stride(1), active_query_count.stride(2),
        # Internal block sizes
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_R=BLOCK_SIZE_R,
        num_warps=4,
        num_stages=3,
    )
    return active_query_count

def reorder_topk_idx_fixedlen(
    topk_idx: torch.Tensor,         # Shape: [batch, num_heads, seq_len, topk]
    active_query_count: torch.Tensor, # Shape: [batch, num_heads, num_blocks]
    block_size: int
) -> torch.Tensor:
    """
    Reorders topk_idx for the backward pass in a fixed-length (batched) setting.
    It transforms the query-centric topk_idx into a key-centric index using the "argsort trick".

    This version correctly generates QUERY BLOCK indices as expected by the backward_dkdv kernel.

    Args:
        topk_idx: Tensor containing key block indices for each query block.
                  Shape: [batch, num_heads, num_q_blocks, topk].
        active_query_count: Tensor from count_query_fixedlen, where [b, h, k]
                            is the number of query blocks in batch b, head h that
                            attend to key block k.
        block_size: The size of the query/key/value blocks.

    Returns:
        A compact tensor `topk_q_idx` where query BLOCK indices are grouped by the
        key block they attend to. The layout is key-centric.
    """
    batch_size, num_kv_heads, num_q_blocks, topk = topk_idx.shape
    num_blocks = active_query_count.shape[-1]
    device = topk_idx.device

    # === Step 1: The "Argsort Trick" to Group Query BLOCK Indices by Key Block ===

    # (a) Flatten the query-block and topk dimensions.
    flat_topk_idx = topk_idx.reshape(batch_size, num_kv_heads, -1)

    # (b) Map padding values (-1) to 'num_blocks' to push them to the end after sorting.
    sort_keys = torch.where(flat_topk_idx < 0, num_blocks, flat_topk_idx)

    # --- THIS IS THE CRITICAL FIX ---
    # (c) Create a tensor of QUERY BLOCK indices, matching the shape of flat_topk_idx.
    # The original code incorrectly used `torch.arange(seq_len)`, creating sequence indices.
    # The backward kernel expects BLOCK indices.
    query_block_indices = torch.arange(num_q_blocks, device=device, dtype=torch.int32)
    query_block_indices = query_block_indices.unsqueeze(-1).expand(-1, topk).reshape(1, 1, -1)
    query_block_indices = query_block_indices.expand(batch_size, num_kv_heads, -1)
    # --- END OF CRITICAL FIX ---

    # (d) Sort the 'sort_keys'. The resulting indices map from the new sorted order
    # back to the original unsorted order.
    sorted_indices = torch.argsort(sort_keys, dim=-1, stable=True)

    # (e) Gather the query block indices using the sorted_indices. The result is `pad_topk_q_idx`,
    # where query BLOCK indices are now grouped by the key block they attend to.
    pad_topk_q_idx = torch.gather(query_block_indices, dim=-1, index=sorted_indices)

    # === Step 2: Compact the Result ===

    # (a) Calculate cumulative counts. This tells the compaction kernel where to write
    # the data for each key block.
    cu_topk_q_count = torch.nn.functional.pad(
        active_query_count.cumsum(dim=-1), (1, 0)
    ).to(torch.int32)

    # (b) Create the final output tensor.
    max_connections_per_sample = cu_topk_q_count[..., -1].max().item()
    topk_q_idx = torch.full(
        (batch_size, num_kv_heads, max_connections_per_sample),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )

    # (c) Launch the compaction kernel.
    grid = (batch_size, num_kv_heads, num_blocks)
    BLOCK_SIZE_M = 256

    save_topk_idx_kernel_fixedlen[grid](
        pad_topk_q_idx,
        topk_q_idx,
        cu_topk_q_count,
        pad_topk_q_idx.stride(0), pad_topk_q_idx.stride(1), pad_topk_q_idx.stride(2),
        topk_q_idx.stride(0), topk_q_idx.stride(1), topk_q_idx.stride(2),
        cu_topk_q_count.stride(0), cu_topk_q_count.stride(1), cu_topk_q_count.stride(2),
        p_size=pad_topk_q_idx.shape[-1],
        t_size=topk_q_idx.shape[-1],
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )

    return topk_q_idx

@triton.jit
def backward_dkdv(
    q_ptr,  # Q: b x h x l x d
    k_ptr,  # K: b x h x l x d
    v_ptr,  # V: b x h x l x d
    tq_ptr,  # topk_q_idx: b x h x (l / block_size) x topk
    cu_tq_cnt_ptr,  # Cumulative TopK query count: b x h x (l / block_size + 1)
    lse_ptr,  # LSE: b x h x l
    delta_ptr,    # Delta: b x h x l
    do_ptr,
    dk_ptr,  # DK: b x h x l x d
    dv_ptr,  # DV: b x h x l x d
    # shape
    seq_len,
    nheads,
    # sm_scale
    sm_scale,
    # num_key_blocks
    num_key_blocks,
    # stride
    stride_qb, stride_qh, stride_ql, 
    stride_kb, stride_kh, stride_kl, 
    stride_vb, stride_vh, stride_vl, 
    stride_tqb, stride_tqh,
    stride_cutqcb, stride_cutqch, # Corrected strides for cumulative tensor
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

    if key_block_id >= num_key_blocks:
        return

    cu_tq_cnt_base_ptr = cu_tq_cnt_ptr + off_b * stride_cutqcb + off_h * stride_cutqch
    start_pos = tl.load(cu_tq_cnt_base_ptr + key_block_id).to(tl.int32)
    end_pos = tl.load(cu_tq_cnt_base_ptr + key_block_id + 1).to(tl.int32)
    tq_cnt = end_pos - start_pos
    
    # Early return if no queries attend to this key block
    if tq_cnt <= 0:
        return
    
    # For sparse attention, the key block position is simply key_block_id * BLOCK_SIZE_K
    key_seq_start = key_block_id * BLOCK_SIZE_K
    
    # Bounds check for key sequence position
    if key_seq_start >= seq_len:
        return


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

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
    d_base = delta_ptr + off_b * stride_db + off_h * stride_dh
    lse_base = lse_ptr + off_b * stride_lb + off_h * stride_lh

    k = tl.load(k_ptrs, boundary_check=(0, 1))
    v = tl.load(v_ptrs, boundary_check=(0, 1))

    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)

    for i in tl.range(tq_cnt):
        query_idx = tl.load(tq_ptr + off_b * stride_tqb + off_h * stride_tqh + start_pos + i).to(tl.int32)
        
        q_seq_start = query_idx * BLOCK_SIZE_Q
        
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
        
        q = tl.load(q_ptrs, boundary_check=(0, 1))
        do = tl.load(do_ptrs, boundary_check=(0, 1))
        
        offs_q = q_seq_start + tl.arange(0, BLOCK_SIZE_Q)
        mask_q = offs_q < seq_len
        lse = tl.load(lse_base + offs_q, mask=mask_q, other=0.0)
        d = tl.load(d_base + offs_q, mask=mask_q, other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = qk_scale * p * (dp - d[:, None])
        
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        
        dk += tl.dot(tl.trans(ds), q)
        dv += tl.dot(tl.trans(p), do)

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
    topk_ptr,  # topk_idx: b x h x (l / block_m) x k
    lse_ptr,  # LSE: B x h x L
    d_ptr,  # Delta: B x h x L
    do_ptr, # dO: b x h x l x d
    dq_ptr, # dQ: b x h x l x d
    # shape
    seq_len,
    nheads,
    TOPK,
    # sm_scale
    sm_scale,
    block_size,
    # stride
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_tb, stride_th, stride_tm,
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
    
    # TopK pointer for this query block
    t_ptrs = topk_ptr + off_b * stride_tb + off_h * stride_th + start_m * stride_tm

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
    for block_idx in tl.range(TOPK):
        # Load the key block index
        topk_idx = tl.load(t_ptrs + block_idx).to(tl.int32)
        k_seq_start = topk_idx * block_size
            
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


def _topk_blocksparse_attention_fwd(
    q: torch.Tensor,  # [batch, nheads, seq_len, HEAD_DIM]
    k: torch.Tensor,  # [batch, nheads, seq_len, HEAD_DIM]
    v: torch.Tensor,  # [batch, nheads, seq_len, HEAD_DIM]
    topk_idx: torch.Tensor,  # [batch, nheads, seq_len, topk]
    block_size: int,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert block_size in {4, 16, 32, 64, 128, 256}
    # shape
    batch, nheads, seq_len, HEAD_DIM = q.shape
    
    ntopk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == batch
    assert topk_idx.shape[1] == nheads
    assert topk_idx.shape[2] == seq_len // block_size

    # N should be identical to block size for compression, let's say both 64 now
    BLOCK_M = block_size
    BLOCK_N = block_size
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.zeros(batch, nheads, seq_len, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * nheads)
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_M, IS_HOPPER_GPU)
    # pdb.set_trace()
    block_sparse_forward_kernel[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        lse,
        sm_scale,
        seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        TOPK=ntopk,
        HEAD_DIM=HEAD_DIM,
        nheads=nheads,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse


def _topk_blocksparse_attention_bwd(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    sm_scale: float,
):
    assert block_size in {16, 32, 64, 128, 256, 512, 1024}
    batch, nheads, seq_len, HEAD_DIM = q.shape
    topk = topk_idx.shape[-1]

    # compute D
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

    # active query idx for each key block
    # how to get active query idx for sequence b, head h, kv block i?
    topk_q_count = count_query_fixedlen(topk_idx, block_size)  # [batch, num_heads, num_blocks]
    topk_q_idx = reorder_topk_idx_fixedlen(topk_idx, topk_q_count, block_size)
    cu_topk_q_count = torch.nn.functional.pad(
        topk_q_count.cumsum(dim=-1), (1, 0)
    ).to(torch.int32)

    # compute dk dv
    dk = torch.zeros(
        batch, nheads, seq_len, HEAD_DIM, device=k.device, dtype=k.dtype
    )
    dv = torch.zeros(
        batch, nheads, seq_len, HEAD_DIM, device=k.device, dtype=k.dtype
    )

    BLOCK_SIZE_K = block_size
    BLOCK_SIZE_Q = block_size
    BLOCK_SIZE_D = triton.next_power_of_2(HEAD_DIM)
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_SIZE_Q, IS_HOPPER_GPU)

    # (num_key_blocks, bs * heads)
    num_key_blocks = topk_q_count.shape[-1]
    grid = (num_key_blocks, batch * nheads)
    
    backward_dkdv[grid](
        q,
        k,
        v,
        topk_q_idx,
        cu_topk_q_count,
        lse,
        delta,
        do,
        dk,
        dv,
        seq_len,
        nheads,
        sm_scale,
        num_key_blocks,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_q_idx.stride(0), topk_q_idx.stride(1),
        cu_topk_q_count.stride(0), cu_topk_q_count.stride(1), # <-- Use strides of the new tensor
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
    BLOCK_M = block_size
    BLOCK_N = block_size
    num_warps, num_stages = get_num_warps_stages(HEAD_DIM, BLOCK_SIZE_K, IS_HOPPER_GPU)

    grid_dq = (triton.cdiv(seq_len, BLOCK_M), batch * nheads)

    backward_dq[grid_dq](
        q,
        k,
        v,
        topk_idx,
        lse,
        delta,
        do,
        dq,
        seq_len,
        nheads,
        topk,
        sm_scale,
        block_size,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
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


class TopkBlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [total_len, num_q_heads, HEAD_DIM]
        k: torch.Tensor,  # [total_len, num_k_heads, HEAD_DIM]
        v: torch.Tensor,  # [total_len, num_k_heads, HEAD_DIM]
        topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
        block_size: int,
        sm_scale=None,
    ):
        # dtype check
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert topk_idx.dtype == torch.int32
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _topk_blocksparse_attention_fwd(
            q,
            k,
            v,
            topk_idx,
            block_size,
            sm_scale,
        )
        ctx.save_for_backward(q, k, v, o, lse, topk_idx)
        ctx.sm_scale = sm_scale
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, topk_idx = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        block_size = ctx.block_size
        assert block_size in {16, 32, 64, 128, 256}

        dq, dk, dv = _topk_blocksparse_attention_bwd(
            o,
            do,
            lse,
            q,
            k,
            v,
            topk_idx,
            block_size,
            sm_scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def topk_blocksparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Topk sparse attention varlen version implemented in triton.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, HEAD_DIM]
        k (torch.Tensor): shape [total_len, num_kv_heads, HEAD_DIM]
        v (torch.Tensor): shape [total_len, num_kv_heads, HEAD_DIM]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(HEAD_DIM).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, HEAD_DIM]
    """
    return TopkBlockSparseAttention.apply(
        q,
        k,
        v,
        topk_idx,
        block_size,
        softmax_scale,
    )

