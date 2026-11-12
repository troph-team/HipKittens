import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); g = H // HKV; scale = D ** -0.5
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.full((1, N, H, D), 7.0, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse); torch.cuda.synchronize()
for h in (0, 1, 8):
    s = (q[0, :, h].float() @ k[0, :, h // g].float().t()) * scale
    s.masked_fill_(torch.ones(N, N, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
    ref = torch.softmax(s, -1) @ v[0, :, h // g].float()
    ref_lse = torch.logsumexp(s, -1)
    oo = o[0, :, h].float()
    per_tile = [((oo[t*32:(t+1)*32] - ref[t*32:(t+1)*32]).norm() / ref[t*32:(t+1)*32].norm()).item() for t in range(N // 32)]
    per_chunk = [((oo[:, c*128:(c+1)*128] - ref[:, c*128:(c+1)*128]).norm() / ref[:, c*128:(c+1)*128].norm()).item() for c in range(D // 128)]
    print(f"head {h}: tiles 0-7 err {[round(x,3) for x in per_tile[:8]]}  last {round(per_tile[-1],3)}  untouched(7.0) {(oo == 7.0).float().mean().item():.2f}")
    print(f"   per D-chunk err {[round(x,3) for x in per_chunk]}  nan rows {oo.isnan().any(-1).sum().item()}  lse err {(lse[0, h, 0] - ref_lse).abs().max().item():.3e}")
