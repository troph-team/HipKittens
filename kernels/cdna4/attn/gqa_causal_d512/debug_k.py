import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); g = H // HKV
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N); torch.cuda.synchronize()
h = 0; ref = k[0, :64, h // g].float(); got = o[0, :64, h].float()
print("K tile0 roundtrip rel err:", ((got[:, :D] - ref).norm() / ref.norm()).item())
print("got[0,:4]", got[0,:4].tolist(), "ref[0,:4]", ref[0,:4].tolist())
# is it a permutation of rows?
d = torch.cdist(got[:, :D], ref)
print("best-match row for got rows 0..7:", d.argmin(1)[:8].tolist(), " min dist", [round(x,2) for x in d.min(1).values[:8].tolist()])
