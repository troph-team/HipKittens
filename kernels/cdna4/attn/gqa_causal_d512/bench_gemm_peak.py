import torch, triton
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
ms = triton.testing.do_bench(lambda: a @ b, warmup=5, rep=50)
print(f"bf16 GEMM 8192^3: {2 * 8192**3 / ms / 1e9:.0f} TFLOP/s")
