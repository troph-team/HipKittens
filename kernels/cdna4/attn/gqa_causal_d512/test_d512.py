"""Correctness and speed of the head_dim-512 GQA causal kernel vs fp32 SDPA and SGLang's Triton extend attention.
Build with matching ATTN_N / ATTN_H / ATTN_H_KV, then: python test_d512.py <N> [H] [H_KV]
"""
import sys, torch, triton
import tk_kernel
N = int(sys.argv[1]); H = int(sys.argv[2]) if len(sys.argv) > 2 else 32; HKV = int(sys.argv[3]) if len(sys.argv) > 3 else 4
import os
D = int(os.environ.get("ATTN_D", "512")); g = H // HKV; scale = D ** -0.5
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16)
lse = torch.zeros(1, H, 1, N, device="cuda", dtype=torch.float32)
run = lambda: tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N, torch.cuda.current_stream().cuda_stream)
run(); torch.cuda.synchronize()
# fp32 reference, a slice of heads to keep memory bounded
errs = []
for h in range(0, H, max(1, H // 4)):
    s = (q[0, :, h].float() @ k[0, :, h // g].float().t()) * scale
    s.masked_fill_(torch.ones(N, N, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
    ref = torch.softmax(s, -1) @ v[0, :, h // g].float()
    errs.append(((o[0, :, h].float() - ref).norm() / ref.norm()).item())
print(f"rel err vs fp32 (heads sampled): max {max(errs):.3e}  nan {o.isnan().any().item()}")
flops = 4 * N * N * H * D / 2
ms = triton.testing.do_bench(run, warmup=3, rep=20)
print(f"HipKittens d512 v0: {ms:.2f} ms  {flops / ms / 1e9:.0f} TFLOP/s")
from sglang.kernels.ops.attention import extend_attention as m
qt, kt, vt, ot = q[0].contiguous(), k[0].contiguous(), v[0].contiguous(), torch.empty_like(q[0])
empty = torch.zeros(1, HKV, D, device="cuda", dtype=torch.bfloat16)
f = lambda: m.extend_attention_fwd(qt, kt, vt, ot, empty, empty, torch.tensor([0, N], device="cuda"), torch.tensor([0, 0], device="cuda", dtype=torch.int32),
                                   torch.zeros(1, dtype=torch.int64, device="cuda"), None, True, None, N, 1.0, 1.0, sm_scale=scale, extend_seq_lens_cpu=[N])
f(); torch.cuda.synchronize()
ms2 = triton.testing.do_bench(f, warmup=3, rep=20)
print(f"SGLang Triton d512:  {ms2:.2f} ms  {flops / ms2 / 1e9:.0f} TFLOP/s   (HK/Triton speedup {ms2 / ms:.2f}x)")
