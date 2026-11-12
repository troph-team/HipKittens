# Gemma 4 31B on MI350P (gfx950): measurements and the head_dim-512 attention port

This is the context for porting HipKittens attention to head_dim 512. Gemma 4 31B's 10 global layers use 32 query
heads over 4 KV heads at head_dim 512. Serving `RedHatAI/gemma-4-31B-it-FP8-block` with SGLang, extend attention in
those layers is the largest prefill cost: 65% of GPU time on a 64K-cached + 32K-new request before tuning, still about
half after it. Full write-up: `troph-team/amd-inference-eval`, worklog 16.

All numbers are on one MI350P, built natively against the pip ROCm SDK (`ROCM_PATH=$(rocm-sdk path --root)`, that
tree's `hipcc`). The system `/opt/rocm` on the box has no HIP headers.

## What HipKittens does today on this card

**FP8 GEMM** (`kernels/cdna4/gemm/fp8fp32/FP8_4wave`, per-tensor FP8, bf16 out, M = 8192). Problem size is set with
`-DGM -DGN -DGK`.

| N×K (Gemma linear) | HipKittens | hipBLASLt rowwise | SGLang Triton block-FP8 (tuned) |
|---|---|---|---|
| 16384×5376 (qkv, sliding) | 1060 TFLOP/s | 1231 | 523 |
| 20480×5376 (qkv, global) | 1075 | 1191 | 562 |
| 43008×5376 (gate+up) | 1083 | 1192 | 498 |
| 5376×8192 (o, sliding) | 1371 | 1265 | 495 |
| 5376×16384 (o, global) | 1219 | 1243 | 518 |
| 5376×21504 (down) | 1403 | 1202 | 488 |

The checkpoint needs 128×128 weight / 1×128 activation block scales. hipBLASLt's block scaling is not available on
ROCm, and adding it here needs a per-K-block partial accumulator. The 4 × (64×64 fp32) accumulator tiles per warp
already use ~256 VGPRs. The RedHatAI FP8-dynamic checkpoint (per-channel / per-token) avoids this by running on
hipBLASLt rowwise.

**GQA causal attention** (`kernels/cdna4/attn/gqa_causal`, bf16, head_dim 128, built with `ATTN_B=1 ATTN_H=32
ATTN_H_KV=4 ATTN_N=16384`; `bench_vs_sglang_triton.py`):

| kernel | time | TFLOP/s | rel err vs fp32 SDPA |
|---|---|---|---|
| HipKittens, D=128 | 3.77 ms | 583 | 2.6e-3 |
| SGLang Triton extend, D=128 | 11.30 ms | 195 | 2.1e-3 |
| SGLang Triton extend, D=256 | 16.45 ms | 267 | |
| SGLang Triton extend, D=512 | 30.54 ms | 288 | |

So HipKittens is 3× the Triton kernel at the same shape. Even 2× at D=512 would halve the global-layer cost.

## Why the D=128 kernel can't simply be instantiated at 512

Per warp (32 queries), at D=512:

| register tile | D=128 | D=512 |
|---|---|---|
| `q_reg` (32×D bf16) + `q_reg_transposed` | 64 VGPR | 256 VGPR |
| `k_reg` (64×D bf16) + transposed | 128 | 512 |
| `o_reg` (32×D fp32) | 64 | 256 |
| `att_block[2]` (64×32 fp32) + bf16 copies | ~80 | ~80 |

The budget is 512 VGPRs + AGPRs per lane at one wave per SIMD. Double-buffered K and V tiles (64×512 bf16 each) need
256 KB of LDS against 160 KB.

## Port plan

1. **Q stays in LDS, not registers.** Stream Q in D-chunks of 128 for QKᵀ: S += Q[:, d:d+128] · K[:, d:d+128]ᵀ, four
   MFMA passes per KV tile. That keeps the existing 128-wide sub-tile layouts and swizzles.
2. **KV block 32 instead of 64**, K and V double-buffered as 32×512 bf16 (32 KB each, 128 KB total). With FP8 KV
   (what the server stores) that halves to 64 KB, and the K tile can be fed to fp8 MFMA directly.
3. **O accumulator in AGPRs** (32×512 fp32 = 256 per lane). P·V runs as four D-chunks writing four 32×128 O slices.
4. **Keep the online softmax, masking and `exp2` scheduling from `gqa_causal`.** Only the QK and PV inner loops change
   shape.
5. **Prefix plus extend.** SGLang calls extend attention with a paged FP8 prefix (`kv_indices`) and a bf16 extend
   chunk. The first version can take a gathered contiguous prefix (the gather is <1 ms per layer). Paging can come
   later.
6. **Accuracy target:** bf16 QK and P scaled before any FP8 PV. SGLang's current FP8-prefix path is 11% off fp32
   (unscaled P cast to FP8). The fixed Triton variant is 0.35%.

Unfused attention as blocked GEMMs (hipBLASLt QKᵀ, separate softmax, PV, LSE merge) was prototyped and is 3× slower
than the Triton kernel (390 vs 131 ms on the real call), so the port has to stay fused.
