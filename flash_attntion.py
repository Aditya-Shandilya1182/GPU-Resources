import torch
import triton
import triton.language as tl
import time
import torch.nn.functional as F
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse, TMP,
    softmax_scale,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_ok,
    stride_lse_m,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1e9, dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk *= softmax_scale

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, 1)

        acc = acc * (l_i / l_ij)[:, None] * tl.exp(m_i - m_ij)[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij
        l_i = l_ij

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

    lse_ptrs = Lse + offs_m * stride_lse_m
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def attention(Q, K, V, causal=False, sm_scale=None):
    N_CTX, D = Q.shape
    sm_scale = sm_scale or (1.0 / math.sqrt(D))

    Out = torch.empty_like(Q)
    Lse = torch.empty((N_CTX,), device=Q.device, dtype=torch.float32)
    TMP = torch.empty((N_CTX,), device=Q.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(N_CTX, BLOCK_M),)

    _fwd_kernel[grid](
        Q, K, V, Out, Lse, TMP,
        sm_scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        Lse.stride(0),
        N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
        IS_CAUSAL=causal,
        num_warps=2, num_stages=1
    )
    return Out


def main():
    seq_len = 128
    dim = 64

    Q = torch.randn((seq_len, dim), device='cuda', dtype=torch.float32)
    K = torch.randn((seq_len, dim), device='cuda', dtype=torch.float32)
    V = torch.randn((seq_len, dim), device='cuda', dtype=torch.float32)

    Q_torch = Q.unsqueeze(0).unsqueeze(0)
    K_torch = K.unsqueeze(0).unsqueeze(0)
    V_torch = V.unsqueeze(0).unsqueeze(0)

    start = time.time()
    out3 = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch, is_causal=True)
    end = time.time()
    print(f"PyTorch attention time: {end - start:.6f}s")
    print(out3.squeeze(0).squeeze(0))

    start = time.time()
    out = attention(Q, K, V, causal=True)
    end = time.time()
    print(f"Triton attention time: {end - start:.6f}s")
    print(out)

if __name__ == "__main__":
    main()
