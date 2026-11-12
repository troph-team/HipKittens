"""HipKittens GQA causal attention vs SGLang's Triton extend attention on gfx950, same shape.

One sequence of N tokens, 32 q heads over 4 kv heads (Gemma 4 GQA ratio), bf16, causal, no prefix. HipKittens supports
head_dim 64/128 only, so the comparison is at head_dim 128; SGLang Triton is also run at head_dim 256 and 512 to show how
much it loses as head_dim grows. Correctness is checked against torch SDPA (fp32 accumulate) for both kernels at 128.
Run inside the SGLang venv from HipKittens/kernels/cdna4/attn/gqa_causal (built with ATTN_B=1 ATTN_H=32 ATTN_H_KV=4
ATTN_N=16384 ATTN_D=128):  python bench_hk_vs_triton_attn.py 16384
"""
import sys, torch, triton
import tk_kernel
from sglang.kernels.ops.attention import extend_attention as m
N = int(sys.argv[1]); HQ, HKV = 32, 4
flops = lambda D: 4 * N * N * HQ * D / 2


def triton_run(D):
    q = torch.randn(N, HQ, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(N, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(N, HKV, D, device="cuda", dtype=torch.bfloat16); o = torch.empty_like(q)
    empty = torch.zeros(1, HKV, D, device="cuda", dtype=torch.bfloat16)
    f = lambda: m.extend_attention_fwd(q, k, v, o, empty, empty, torch.tensor([0, N], device="cuda"),
                                       torch.tensor([0, 0], device="cuda", dtype=torch.int32), torch.zeros(1, dtype=torch.int64, device="cuda"),
                                       None, True, None, N, 1.0, 1.0, sm_scale=D ** -0.5, extend_seq_lens_cpu=[N])
    f(); torch.cuda.synchronize()
    return f, q, k, v, o


torch.manual_seed(0)
q = torch.randn(1, N, HQ, 128, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, 128, device="cuda", dtype=torch.bfloat16)
out = torch.zeros(1, N, HQ, 128, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, HQ, 1, N, device="cuda", dtype=torch.float32)
hk = lambda: tk_kernel.dispatch_micro(q, k, v, out, lse)
hk(); torch.cuda.synchronize()
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float().repeat_interleave(8, 1),
                                                       v.transpose(1, 2).float().repeat_interleave(8, 1), is_causal=True).transpose(1, 2)
ms = triton.testing.do_bench(hk, warmup=5, rep=50)
print(f"HipKittens   D=128: {ms:7.2f} ms {flops(128) / ms / 1e9:6.0f} TFLOP/s  rel_err {((out.float() - ref).norm() / ref.norm()).item():.2e}")
f, tq, tk, tv, to = triton_run(128)
tq.copy_(q[0]); tk.copy_(k[0]); tv.copy_(v[0]); f(); torch.cuda.synchronize()
ms = triton.testing.do_bench(f, warmup=5, rep=50)
print(f"SGLang Triton D=128: {ms:7.2f} ms {flops(128) / ms / 1e9:6.0f} TFLOP/s  rel_err {((to.float() - ref[0]).norm() / ref[0].norm()).item():.2e}")
for D in (256, 512):
    f, *_ = triton_run(D)
    ms = triton.testing.do_bench(f, warmup=3, rep=20)
    print(f"SGLang Triton D={D}: {ms:7.2f} ms {flops(D) / ms / 1e9:6.0f} TFLOP/s")
