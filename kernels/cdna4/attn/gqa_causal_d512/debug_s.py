import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); g = H // HKV; sc = D ** -0.5 * 1.44269504089
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N); torch.cuda.synchronize()
h = 0
S = (q[0, :, h].float() * sc) @ k[0, :64, h // g].float().t()   # (N q, 64 kv)
got = o[0, :, h, :64].float()
for t in (0, 1, 5):
    r = S[32*t:32*t+32]; gg = got[32*t:32*t+32]
    print(f"q tile {t}: rel err {((gg - r).norm() / r.norm()).item():.3e}   got[0,:3] {[round(x,3) for x in gg[0,:3].tolist()]}  ref[0,:3] {[round(x,3) for x in r[0,:3].tolist()]}")
