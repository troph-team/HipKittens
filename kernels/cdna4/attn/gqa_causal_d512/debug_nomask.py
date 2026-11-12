import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); g = H // HKV; scale = D ** -0.5
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N); torch.cuda.synchronize()
h = 0
for t in (0, 1, 2, 5, 9):
    nk = ((32 * t + 32 + 63) // 64) * 64   # keys this warp processes (whole KV tiles, no mask)
    s = (q[0, 32*t:32*t+32, h].float() @ k[0, :nk, h // g].float().t()) * scale
    ref = torch.softmax(s, -1) @ v[0, :nk, h // g].float()
    got = o[0, 32*t:32*t+32, h].float()
    print(f"q tile {t} (keys 0..{nk}): rel err {((got - ref).norm() / ref.norm()).item():.3e}  nan {got.isnan().any().item()}")
