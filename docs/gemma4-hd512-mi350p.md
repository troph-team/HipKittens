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

## Status of `kernels/cdna4/attn/gqa_causal_d512` (correct; 1.14× SGLang Triton at N = 16K)

v0 is a straightforward kernel with no double buffering and no hand scheduling. Per 128-wide D chunk it reuses the
D=128 kernel's tile types: S^T = Σ_c K_c^T·Q_c^T and O_c^T += V_c·P^T, with K and V held as 4 separate 64×128 shared
tiles each. It runs 4 warps per block and stages K/V through registers.

| N (1 seq, 32 q / 4 kv heads, causal) | HipKittens d512 v0 | SGLang Triton d512 | rel err vs fp32 |
|---|---|---|---|
| 2048 | 0.67 ms, 206 TFLOP/s | 0.68 ms, 202 TFLOP/s | 2.5e-3 |
| 16384 | 29.1 ms, 302 TFLOP/s | 30.6 ms, 288 TFLOP/s | 2.6e-3 |

The same kernel built at D=128 matches the original `gqa_causal` (0.2–0.3% error).

Optimizations so far (D=512, N=16384; build flags in parentheses):

| version | spill | time | TFLOP/s | vs Triton (30.6 ms) |
|---|---|---|---|---|
| v0: KV block 64, register-staged | 126 VGPR | 29.1 ms | 302 | 1.05× |
| KV block 32 (`-DKV_BLOCK=32`) | 10 | 27.8 ms | 316 | 1.10× |
| + direct-to-LDS, double-buffered K/V (`-DDIRECT_LDS -DDOUBLE_BUF -DKV_BLOCK=32`) | 15 | 26.8 ms | 329 | 1.14× |

**Prefix + extend** (runtime `q_off`, `n_q`, `n_kv`; `test_extend.py`; `-DATTN_SCALE=1.0f` for Gemma 4).
- Correct vs fp32: 2.8e-3 at P=2048 / E=512, and 2.9e-3 at P=32000 / E=1024, a prefix that isn't a multiple of the KV
  block.
- At Gemma's replay shape (P=65536 cached + E=8192 new):

| kernel | time | TFLOP/s | rel err vs fp32 |
|---|---|---|---|
| **HipKittens d512, bf16 prefix** | **129.5 ms** | **289** | **0.28%** |
| SGLang Triton extend, bf16 prefix | 184.2 ms | 203 | |
| SGLang Triton extend, FP8 prefix + hd512 tile (server today, from amd-inference-eval worklog 16) | 131 ms | | 11.2% |
| SGLang Triton extend, FP8 prefix + tile + `SGLANG_FP8_PREFIX_PRECISE` | 144 ms | | 0.35% |

So HipKittens is 1.42× Triton at equal precision (bf16 prefix). Against the server's FP8 path it matches the fast
variant's speed at 40× its accuracy, and beats the accurate variant by 1.11×. Feeding it from SGLang's paged FP8
cache means gathering and upcasting the prefix to bf16 per layer, ~0.4 GB of traffic or well under 1 ms.

**In SGLang** (amd-inference-eval `patches/sglang_hk_d512_extend.py`, `SGLANG_HK_D512=1`). The route gathers the paged
FP8 prefix, upcasts it to bf16, and calls this kernel for Gemma 4's global layers. Measured on the optimized
FP8-block stack (split K/V allocation, tuned GEMMs):

| case | Triton FP8 path (hd512 tile) | Triton + precision fix | HipKittens route |
|---|---|---|---|
| extend 64K + 32K | 11.76 s | 12.35 s | 12.18 s |
| cold 64K | 15.8 s | 16.2 s | 16.07 s |
| global-layer attention vs fp32 | ~11% | 0.35% | 0.28% |

Greedy output on 6 replay prompts is fluent and on-format. Per call the kernel is level with Triton's FP8 path:
127 ms of kernel time plus 2.4 ms to gather and upcast. So the gain today is accuracy, not speed. The next step for
speed is to read FP8 K/V from the cache directly: half the LDS and global traffic, and FP8 MFMA for P·V with scaled P.

Integration bugs, both fixed:
1. **Kernel launched on the null stream.** Now `stream_ptr` = `torch.cuda.current_stream().cuda_stream`.
2. **The prefix length came from `kv_indptr[1].item()`.** It reads a tensor that SGLang fills asynchronously, so it
   sometimes got a stale value and generation became garbage. Now it uses the host-known `kv_indices.numel()`.

Tried and rejected: streaming Q chunks from global per KV tile (`-DQ_STREAM`). It removes all spills but is slower
(39.8 ms), because the per-chunk waits expose latency.

Bugs found on the way:
1. **`G::load(st, gl, coord)` defaults to axis 2.** For (B, N, H, D) tensors pass `<1, false>`.
2. **The direct-to-LDS `G::load` (swizzled offsets + srsrc) looked broken at 4 warps, but wasn't.** It was the same
   bug as 3. With the fences it is exact at 4 warps.
3. **Fence every manual `s_waitcnt` with `__builtin_amdgcn_sched_barrier(0)`.** This was the NaN/garbage bug.
   HipKittens' shared→register loads are inline-asm `ds_read`s the compiler doesn't track, so without the fence it
   hoists consumers (e.g. the K transpose) above `s_waitcnt lgkmcnt(0)` and reads registers before the data lands.
   Late data (KV rows 32–63) came out wrong.
4. **`-ffast-math` is finite-math-only,** so the running max starts at -1e30, not -inf.
5. **Debug probes that don't work:** P = 1 via `zero` + `add(tile, 1.0f)` doesn't give a usable all-ones P (the
   original kernel fails the same probe), and storing an S^T tile shows store artifacts on some tiles in both kernels.

Next: remove the 126 spilled VGPRs (e.g. stream Q chunks instead of holding all 4 in registers), double-buffer K/V,
then interleave MFMA with softmax as `gqa_causal` does. After that, a prefix + extend entry point for SGLang.

## v2 (lazy rescale, 64-wide chunks): 129 → 99 ms at P = 64K, E = 8K

Build: `-DDIRECT_LDS -DDOUBLE_BUF -DKV_BLOCK=32 -DDC_SIZE=64 -DLAZY_RESCALE -DQK_ABT -DATTN_SCALE=1.0f`, 4 warps.

**Timing probes**, each removing one part (memory loads removed in all rows but the first two):

| removed | v1 | v2 |
|---|---|---|
| nothing (memory on) | 129.3 ms | 99.4 ms |
| global K/V loads | 102.8 | 98.7 |
| V LDS read + P·V MFMA | 47.2 | 50.6 |
| P·V MFMA only | 59.9 | 67.1 |
| QK MFMA only | 87.3 | 84.5 |
| both MFMAs | 51.6 | 61.1 |
| the O rescale | 78.8 | (lazy) |

- **Lazy rescale** keeps a stale running max unless a query's max grows by more than 2^8 (warp ballot). It is exact.
- **DC = 64** removes the 14–24 VGPR spill.
- **What's left:**
  - Memory is gone as a cost.
  - About 61 ms is LDS reads, softmax and barriers that never overlap the ~38 ms of MFMA, because there is one
    wave per SIMD.
  - Pipelining the LDS reads inside a wave gives ≤ 2%. That figure is with `KITTENS_LDS_TR_BUILTIN`: the inline-asm
    reads are invisible to the compiler, which moved registers whose reads were still in flight.

**In SGLang** (FP8-dynamic, 64K cached + 32K new): this kernel is still 58% of all GPU time in the prefill (4.86 of
8.45 s, 50 calls). FP8 GEMMs are about 18% and sliding-window attention 6%.

**Stale-LDS NaN.** When `n_kv` is not a multiple of KV_BLOCK, the rows past the end of the buffer are never written
into LDS. The last block then holds whatever an earlier kernel left there, and P = 0 times a stale NaN is NaN. The
SGLang wrapper pads K/V to whole blocks. A kernel-side fix would zero those LDS rows, or zero V in registers past
`n_kv`.

## Next: two waves per SIMD (warp pairs split head_dim)

O at 32 queries × 512 is 256 fp32 registers, so a second wave per SIMD cannot fit unless head_dim is split. The plan:

- **Warps:** 8 per block, as 4 query tiles × 2 head_dim halves. Each wave holds Q[32×256] bf16 (64 VGPRs) and
  O[256×32] fp32 (128 AGPRs), within a 256-register budget.
- **Per KV tile:**
  1. Each wave computes a partial Sᵀ over its 256 dims.
  2. The pair swaps partials through LDS (4 KB per warp) and adds them.
  3. Both run the same softmax, so the lazy-rescale decision is identical.
  4. Each wave does P·V on its own V half.
- **LDS reads per tile** are unchanged in total (each wave reads half of K and half of V). The gain is that one
  wave's LDS reads and softmax can run under the other wave's MFMAs; `s_setprio` ping-pong as in the D=128 kernel.
- **Upper bound:** max(61, 38) ms plus the swap, against 99 ms now.

**Considered and set aside:**
- **K rebuilt from V.** Gemma 4's global layers have `attention_k_eq_v`, with partial rotary 0.25. So the 384
  un-rotated K dims are exactly `k_norm.weight ⊙ V`, and folding the weight into Q would let the kernel load only
  K_rot (128 dims) plus V. That cuts K/V memory traffic 37.5%, but QK still needs V read in a second register
  layout, so LDS read traffic doesn't drop, and memory is no longer the limit.
- **FP8 P·V.** HipKittens' FP8 register tiles cover only the 16×16×128 MFMA, which needs 128-row KV blocks and
  doesn't fit in LDS. The 32×32×64 form would need raw MFMA calls and a custom V layout.

### Result: two waves per SIMD (`kernel_split.cpp`) is slower

- **Built as planned:** 8 warps, pairs split head_dim, partial Sᵀ swapped through LDS. It uses the full 163,840-byte
  LDS, against HipKittens' 160,000-byte default. Correct first time (0.29% error).
- **Speed:** 114 ms at P = 64K, E = 8K, against 99.4 ms for v2. Occupancy is 2 waves per SIMD and 256 VGPRs, with
  3–9 spilled.
- **The swap is the cost.** Without it (wrong results, `-DDEBUG_NOSWAP`) the kernel runs at 91 ms, no spills.
  128-bit LDS accesses for the swap don't change the time; the extra block-wide barrier per KV tile does. With
  16 blocks per CU × 2304 KV tiles, that comes to about 1,500 cycles per barrier.
- **Why it can't win:** barriers keep all 8 waves in the same phase, so the overlap the design relies on doesn't
  happen. Even with a free swap the gain is 8%.
- **What would be needed:** either no block-wide sync between the pair (a spin-wait on an LDS flag, which competes
  with the partner on the same SIMD), or a software pipeline across KV tiles (P·V of tile j-1 while tile j's
  partial S is swapped). The second needs a third K/V buffer, and LDS has no room. Not pursued; v2 stays the
  production kernel.
