"""DEBUG_PV build: O[q, :] should equal sum_{kv=0..63} V[kv, :] for every query (P = 1 on KV tile 0)."""
import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); g = H // HKV
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N); torch.cuda.synchronize()
for h in (0, 9):
    ref = v[0, :64, h // g].float().sum(0)            # (D,)
    got = o[0, :64, h].float()                        # first 2 query tiles
    print(f"head {h}: rel err {((got - ref).norm() / (ref.norm() * got.shape[0] ** 0.5)).item():.3e}  got[0,:4] {got[0,:4].tolist()}  ref[:4] {ref[:4].tolist()}")
