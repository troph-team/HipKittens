"""Prefix + extend: E new queries (positions P .. P+E-1) over P cached + E new keys, head_dim 512, 32 q / 4 kv heads.
Correctness vs fp32 on a small case, then timing at Gemma's replay shape vs SGLang's Triton extend attention (bf16 prefix).
python test_extend.py
"""
import torch, triton
import tk_kernel
from sglang.kernels.ops.attention import extend_attention as m
H, HKV, D = 32, 4, 512; g = H // HKV; scale = 1.0   # Gemma 4: sm_scale 1.0 (build with -DATTN_SCALE=1.0f)


def make(P, E, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, E, H, D, device="cuda", dtype=torch.bfloat16) * D ** -0.25
    k = torch.randn(1, P + E, HKV, D, device="cuda", dtype=torch.bfloat16) * D ** -0.25
    v = torch.randn(1, P + E, HKV, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def hk(q, k, v, P, E):
    o = torch.empty(1, E, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.empty(1, H, 1, E, device="cuda")
    tk_kernel.dispatch_d512(q, k, v, o, lse, P, E, P + E)
    return o


for P, E in ((2048, 512), (1000 * 32, 1024)):   # second: prefix not a multiple of the KV block
    q, k, v = make(P, E)
    o = hk(q, k, v, P, E); torch.cuda.synchronize()
    errs = []
    for h in (0, 9, 31):
        s = (q[0, :, h].float() @ k[0, :, h // g].float().t()) * scale
        mask = torch.ones(E, P + E, dtype=torch.bool, device="cuda"); mask[:, P:] = torch.ones(E, E, dtype=torch.bool, device="cuda").tril()
        s.masked_fill_(~mask, float("-inf"))
        ref = torch.softmax(s, -1) @ v[0, :, h // g].float()
        errs.append(((o[0, :, h].float() - ref).norm() / ref.norm()).item())
    print(f"P={P} E={E}: rel err vs fp32 max {max(errs):.3e}  nan {o.isnan().any().item()}")

P, E = 65536, 8192
q, k, v = make(P, E)
flops = 4 * H * D * E * (P + E / 2)
ms = triton.testing.do_bench(lambda: hk(q, k, v, P, E), warmup=3, rep=20)
print(f"HipKittens d512 extend P={P} E={E}: {ms:.1f} ms  {flops / ms / 1e9:.0f} TFLOP/s")
qt, ke, ve = q[0].contiguous(), k[0, P:].contiguous(), v[0, P:].contiguous()
kb, vb = k[0, :P].contiguous(), v[0, :P].contiguous(); ot = torch.empty_like(qt)
f = lambda: m.extend_attention_fwd(qt, ke, ve, ot, kb, vb, torch.tensor([0, E], device="cuda"), torch.tensor([0, P], device="cuda", dtype=torch.int32),
                                   torch.arange(P, device="cuda"), None, True, None, E, 1.0, 1.0, sm_scale=scale, extend_seq_lens_cpu=[E])
f(); torch.cuda.synchronize()
ms2 = triton.testing.do_bench(f, warmup=3, rep=20)
print(f"SGLang Triton extend (bf16 prefix):     {ms2:.1f} ms  {flops / ms2 / 1e9:.0f} TFLOP/s   HK/Triton {ms2 / ms:.2f}x")
o = hk(q, k, v, P, E)
print("HK vs Triton output rel diff:", ((o[0].float() - ot.float()).norm() / ot.float().norm()).item())
