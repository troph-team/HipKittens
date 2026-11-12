import os, sys, torch
import tk_kernel
N = int(sys.argv[1]); H, HKV = 32, 4; D = int(os.environ.get("ATTN_D", "512")); sc = D ** -0.5 * 1.44269504089
torch.manual_seed(0)
q = torch.randn(1, N, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, N, HKV, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(1, N, H, D, device="cuda", dtype=torch.bfloat16); lse = torch.zeros(1, H, 1, N, device="cuda")
tk_kernel.dispatch_d512(q, k, v, o, lse, 0, N, N); torch.cuda.synchronize()
ref = q.float() * sc
print("Q roundtrip rel err (all heads, rows):", ((o.float() - ref).norm() / ref.norm()).item(), " tile0 head0:", ((o[0,:32,0].float()-ref[0,:32,0]).norm()/ref[0,:32,0].norm()).item())
