"""Timing only: HK d512 extend at P=64K, E=8K, and SGLang Triton with bf16 and fp8 prefixes."""
import torch, triton, sys
import tk_kernel
from sglang.kernels.ops.attention import extend_attention as m
H, HKV, D = 32, 4, 512
P, E = int(sys.argv[1]) if len(sys.argv) > 1 else 65536, int(sys.argv[2]) if len(sys.argv) > 2 else 8192
torch.manual_seed(0)
q = torch.randn(1, E, H, D, device="cuda", dtype=torch.bfloat16) * D ** -0.25
k = torch.randn(1, P + E, HKV, D, device="cuda", dtype=torch.bfloat16) * D ** -0.25
v = torch.randn(1, P + E, HKV, D, device="cuda", dtype=torch.bfloat16)
flops = 4 * H * D * E * (P + E / 2)
o = torch.empty(1, E, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.empty(1, H, 1, E, device="cuda")
hk = lambda: tk_kernel.dispatch_d512(q, k, v, o, lse, P, E, P + E, torch.cuda.current_stream().cuda_stream)
ms = triton.testing.do_bench(hk, warmup=3, rep=20)
print(f"HK  {ms:7.1f} ms  {flops / ms / 1e9:4.0f} TFLOP/s")
if "--hk-only" in sys.argv: sys.exit()
qt, ke, ve = q[0].contiguous(), k[0, P:].contiguous(), v[0, P:].contiguous(); ot = torch.empty_like(qt)
for name, dt in (("bf16", torch.bfloat16), ("fp8", torch.float8_e4m3fn)):
    kb, vb = k[0, :P].to(dt).contiguous(), v[0, :P].to(dt).contiguous()
    f = lambda: m.extend_attention_fwd(qt, ke, ve, ot, kb, vb, torch.tensor([0, E], device="cuda"), torch.tensor([0, P], device="cuda", dtype=torch.int32),
                                       torch.arange(P, device="cuda"), None, True, None, E, 1.0, 1.0, sm_scale=1.0, extend_seq_lens_cpu=[E])
    f(); torch.cuda.synchronize()
    ms2 = triton.testing.do_bench(f, warmup=3, rep=20)
    print(f"Triton {name} prefix {ms2:7.1f} ms  {flops / ms2 / 1e9:4.0f} TFLOP/s")
