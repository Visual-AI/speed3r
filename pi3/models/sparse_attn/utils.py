# Copyright 2025 Xunhao Lai & Jianqiao Lu.
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

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Implements argsort based on bitonic sort.
# [What is bitonic sort?](https://en.wikipedia.org/wiki/Bitonic_sorter)

# Code adapted from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396

import torch
import triton
import triton.language as tl

def is_hopper_gpu():
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        return major == 9
    return False


def get_compressed_seqlens(
    cu_seqlens: torch.Tensor, kernel_size: int, kernel_stride: int
):
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.zeros(
        y_seqlens.shape[0] + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    y_cu_seqlens[1:] = torch.cumsum(y_seqlens, dim=0)
    return y_seqlens, y_cu_seqlens


def get_num_warps_stages(head_dim, block_size, is_hopper_gpu):
    """
    Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

    Args:
        head_dim (int): Size of the head dimension.
        block_size (int): Size of the block in the attention matrix.
        is_hopper_gpu (bool): True if Hopper GPU, False if Ampere GPU.

    Returns:
        tuple: (num_warps, num_stages) recommended values.
    """
    # Determine if head_dim and block_size exceed 64
    head_large = head_dim > 64
    block_large = block_size > 64
    if is_hopper_gpu:
        # Hopper GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    else:
        # Ampere GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    if head_dim > 128:
        num_stages = 2
    return num_warps, num_stages


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(
    x,
    ids,
    dim: tl.constexpr = None,
    descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = tl.log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids