// GQA causal attention forward for head_dim 512 (Gemma 4 global layers) on CDNA4 (gfx950).
//
// v0: correctness first. Same tile types as gqa_causal/kernel.cpp (D = 128), applied per 128-wide chunk of D:
//   S^T (64 kv x 32 q)  = sum_c  K_c^T (128 x 64)^T . Q_c^T (128 x 32)     -- 4 MFMA passes per KV tile
//   O_c^T (128 x 32)   += V_c (64 x 128)^T . P^T (64 x 32)                 -- 4 output chunks, fp32
// Each warp owns 32 queries; NUM_WARPS warps of the same head share one K and one V tile (64 x 512 bf16 each) in LDS.
// No double buffering or hand scheduling yet; online softmax is exact (no rescale threshold).
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 1;
#endif
#ifndef ATTN_H
constexpr int ATTN_H = 32;
#endif
#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 4;
#endif
#ifndef ATTN_N
constexpr int ATTN_N = 4096;
#endif
#ifndef ATTN_D
constexpr int ATTN_D = 512;
#endif
#ifndef ATTN_SCALE  // softmax scale; Gemma 4 uses 1.0 (q/k are RMS-normed), the test default is 1/sqrt(D)
#define ATTN_SCALE (1.0f / __builtin_sqrtf((float)ATTN_D))
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
constexpr int Q_BLOCK_SIZE = 32;
#ifndef KV_BLOCK
#define KV_BLOCK 64
#endif
constexpr int KV_BLOCK_SIZE = KV_BLOCK;
#ifndef DC_SIZE
#define DC_SIZE 128
#endif
constexpr int DC = DC_SIZE;          // head_dim chunk (64 halves the K working registers)
constexpr int NC = ATTN_D / DC;      // number of chunks
static_assert(ATTN_D % DC == 0, "head_dim must be a multiple of the chunk");

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    static_assert(std::is_same_v<shape, rt_32x32_s>, "Only 32x32 tiles are supported");
    #pragma unroll
    for (int k = 0; k < dst.packed_per_thread; k++) dst.data[k] = base_ops::exp2::op(src.data[k]);
}

template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using qo_tile = rt<T, Q_BLOCK_SIZE, DC, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using qo_tile_transposed = rt<T, DC, Q_BLOCK_SIZE, L, S>;
template<typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using kv_tile = rt<T, KV_BLOCK_SIZE, DC, L, S>;
template<typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using kv_tile_transposed = rt<T, DC, KV_BLOCK_SIZE, L, S>;
template<typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

// ---- causal mask on an S^T tile (64 kv rows x 32 q cols), identical to gqa_causal/kernel.cpp ----
template<int THR_X, int THR_Y>
__device__ inline void mask_vec2_imm(uint32_t rel_vgpr, uint32_t neg_inf_vgpr, uint32_t& x_ref, uint32_t& y_ref) {
    uint64_t x_mask, y_mask;
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr), "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc");
}

template<ducks::rt::col_layout RT>
__device__ inline void mask_kv_tile(RT &dst, int q_base_pos, int k_abs, uint32_t neg_inf_v, int lane) {
    const int col = lane & 31;
    const int q_pos = q_base_pos + col;   // absolute position of this lane's query (prefix offset included)
    const int k_base = k_abs * KV_BLOCK_SIZE;
    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        const int row_base = (i * 32) + ((lane >> 5) << 2);
        const uint32_t rel = static_cast<uint32_t>(q_pos - (k_base + row_base));
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].x);
            auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].y);
            auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].x);
            auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].y);
            auto& d2x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].x);
            auto& d2y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].y);
            auto& d3x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].x);
            auto& d3y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].y);
            auto& d4x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].x);
            auto& d4y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].y);
            auto& d5x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].x);
            auto& d5y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].y);
            auto& d6x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].x);
            auto& d6y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].y);
            auto& d7x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].x);
            auto& d7y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].y);
            mask_vec2_imm< 0, 1 >(rel, neg_inf_v, d0x, d0y);
            mask_vec2_imm< 2, 3 >(rel, neg_inf_v, d1x, d1y);
            mask_vec2_imm< 8, 9 >(rel, neg_inf_v, d2x, d2y);
            mask_vec2_imm<10,11 >(rel, neg_inf_v, d3x, d3y);
            mask_vec2_imm<16,17 >(rel, neg_inf_v, d4x, d4y);
            mask_vec2_imm<18,19 >(rel, neg_inf_v, d5x, d5y);
            mask_vec2_imm<24,25 >(rel, neg_inf_v, d6x, d6y);
            mask_vec2_imm<26,27 >(rel, neg_inf_v, d7x, d7y);
        }
    }
}

struct attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;          // (B, N, H, D) and (B, N, H_KV, D), bf16
    gl<float, -1, -1, -1, -1> L_vec;   // (B, H, 1, n_q) natural-log LSE
    // prefix + extend: queries are positions q_off .. q_off + n_q - 1 of a sequence whose K/V hold n_kv rows
    // (n_kv = q_off + n_q for one extend chunk over a cached prefix; q_off = 0, n_kv = n_q is plain causal attention)
    int q_off, n_q, n_kv;
    uint64_t stream_ptr;   // torch.cuda.current_stream().cuda_stream (0 = null stream)
    dim3 grid() { return dim3(ATTN_H, ((n_q + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_ker_d512(const attn_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K and V tiles as NC separate 64 x 128 shared tiles (the exact tile types gqa_causal uses at D = 128), 128 KB total
#ifdef DOUBLE_BUF   // two K/V buffers (needs DIRECT_LDS: async loads); index [buf * NC + chunk]
    constexpr int NBUF = 2;
#else
    constexpr int NBUF = 1;
#endif
    st_bf<KV_BLOCK_SIZE, DC, st_32x32_s> (&k_smem)[NBUF * NC] = al.allocate<st_bf<KV_BLOCK_SIZE, DC, st_32x32_s>, NBUF * NC>();
    st_bf<KV_BLOCK_SIZE, DC, st_8x32_s> (&v_smem)[NBUF * NC] = al.allocate<st_bf<KV_BLOCK_SIZE, DC, st_8x32_s>, NBUF * NC>();

    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();   // this warp's 32-query tile
    const int lane = laneid();
#ifdef DEBUG_ADDR
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        printf("[addr] k0 %u k_end %u v0 %u sizeof(st_k) %u sizeof(st_v) %u\n", (unsigned)(uintptr_t)&k_smem[0].data[0],
               (unsigned)(uintptr_t)&k_smem[NC - 1].data[0] + (unsigned)sizeof(k_smem[0]), (unsigned)(uintptr_t)&v_smem[0].data[0],
               (unsigned)sizeof(k_smem[0]), (unsigned)sizeof(v_smem[0]));
#endif
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;
    const bool warp_active = q_start_pos < g.n_q;
    const int q_pos0 = g.q_off + q_start_pos;   // absolute position of this warp's first query
    const uint32_t neg_inf_v = 0xff800000;

    // K/V staging exactly as gqa_causal/kernel.cpp: direct-to-LDS buffer loads with precomputed swizzled offsets
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
#ifdef DEBUG_NOMEM   // timing probe: every K/V load is out of range (returns 0 without touching memory)
    i32x4 k_srsrc = make_srsrc(k_base, 0, k_row_stride);
    i32x4 v_srsrc = make_srsrc(v_base, 0, v_row_stride);
#else
    i32x4 k_srsrc = make_srsrc(k_base, k_row_stride * g.n_kv, k_row_stride);   // rows past n_kv read as 0 (then masked)
    i32x4 v_srsrc = make_srsrc(v_base, v_row_stride * g.n_kv, v_row_stride);
#endif
    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds[NBUF * NC], v_lds[NBUF * NC];
    #pragma unroll
    for (int c = 0; c < NBUF * NC; c++) {
        k_lds[c] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&k_smem[c].data[0]) + wid * elem_per_warp * sizeof(bf16)));
        v_lds[c] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&v_smem[c].data[0]) + wid * elem_per_warp * sizeof(bf16)));
    }
    using T = typename st_bf<KV_BLOCK_SIZE, DC, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * DC * sizeof(T) / (bytes_per_thread * NUM_THREADS);
    uint32_t swz_K[memcpy_per_tile], swz_V[memcpy_per_tile];   // same offsets serve every chunk (same tile type)
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swz_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swz_V);

    // KV tiles this block needs (causal): up to the last query of its last warp
    const int last_q = g.q_off + min((block_tile_idx + 1) * NUM_WARPS * Q_BLOCK_SIZE, g.n_q);
    const int num_kv_tiles = min((last_q + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE, (g.n_kv + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE);
    // KV tiles this warp needs
    const int my_kv_tiles = (q_pos0 + Q_BLOCK_SIZE + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    // ---- Q: 4 chunks of 32 x 128, pre-scaled by softmax_scale * log2(e), kept transposed in registers ----
#ifndef Q_STREAM
    qo_tile_transposed<bf16> q_t[NC];
    if (warp_active) {
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            qo_tile<float> q_fl;
            load<1, qo_tile<float>, _gl_QKVO>(q_fl, g.Qg, {batch_idx, tile_idx, head_idx, c});
            mul(q_fl, q_fl, ATTN_SCALE * 1.44269504089f);
#ifdef DEBUG_Q
            store<1>(g.Og, q_fl, {batch_idx, tile_idx, head_idx, c});
            continue;
#endif
            qo_tile<bf16> q_bf;
            copy(q_bf, q_fl);
            transpose(q_t[c], q_bf);
        }
    }
#endif

#ifdef DEBUG_Q
    return;
#endif
    qo_tile_transposed<float, col_l, rt_32x32_s> o[NC];
    #pragma unroll
    for (int c = 0; c < NC; c++) zero(o[c]);
    attn_tile<float, col_l, rt_32x32_s> att;
    attn_tile<bf16, col_l, rt_32x32_s> att_bf16;
    attn_tile<bf16, col_l, rt_16x32_4_s> att_in;
    typename attn_tile<float, col_l, rt_32x32_s>::row_vec max_vec, max_prev, norm_vec, scale_vec;
    // finite "minus infinity": the build uses -ffast-math (finite-math-only), so -inf arithmetic is not safe here
    zero(max_prev);
    add(max_prev, max_prev, -1.0e30f);
    zero(norm_vec);

#ifdef DOUBLE_BUF
    constexpr int LOADS_PER_TILE = 2 * NC * memcpy_per_tile;   // buffer_load_lds instructions per K+V tile (vmcnt units)
    static_assert(LOADS_PER_TILE <= 63, "vmcnt field is 6 bits");
    #pragma unroll
    for (int c = 0; c < NC; c++) {   // prologue: KV tile 0 -> buffer 0
        G::load<1, false>(k_smem[c], g.Kg, {batch_idx, 0, head_idx_kv, c}, swz_K, k_srsrc, k_base, k_lds[c]);
        G::load<1, false>(v_smem[c], g.Vg, {batch_idx, 0, head_idx_kv, c}, swz_V, v_srsrc, v_base, v_lds[c]);
    }
#endif
    for (int j = 0; j < num_kv_tiles; j++) {
#ifdef DOUBLE_BUF
        const int cur = (j & 1) * NC;
        // everyone is done with tile j-1, whose buffer tile j+1 now overwrites
        __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        if (j + 1 < num_kv_tiles) {
            const int nxt = ((j + 1) & 1) * NC;
            #pragma unroll
            for (int c = 0; c < NC; c++) {
                G::load<1, false>(k_smem[nxt + c], g.Kg, {batch_idx, j + 1, head_idx_kv, c}, swz_K, k_srsrc, k_base, k_lds[nxt + c]);
                G::load<1, false>(v_smem[nxt + c], g.Vg, {batch_idx, j + 1, head_idx_kv, c}, swz_V, v_srsrc, v_base, v_lds[nxt + c]);
            }
            // wait for tile j only; tile j+1's LOADS_PER_TILE loads stay in flight
            __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(%0)" :: "n"(LOADS_PER_TILE)); __builtin_amdgcn_sched_barrier(0);
        } else {
            __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
#else
        constexpr int cur = 0;
        // all warps cooperatively stage K_j and V_j into LDS
        __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int c = 0; c < NC; c++) {
#ifdef DIRECT_LDS   // HipKittens direct-to-LDS staging: only correct with 8 warps per block
            G::load<1, false>(k_smem[c], g.Kg, {batch_idx, j, head_idx_kv, c}, swz_K, k_srsrc, k_base, k_lds[c]);
            G::load<1, false>(v_smem[c], g.Vg, {batch_idx, j, head_idx_kv, c}, swz_V, v_srsrc, v_base, v_lds[c]);
#else               // register-staged (global -> VGPR -> ds_write), works for any warp count
#ifndef DEBUG_NO_K
            G::load<1, false>(k_smem[c], g.Kg, {batch_idx, j, head_idx_kv, c});
#endif
            G::load<1, false>(v_smem[c], g.Vg, {batch_idx, j, head_idx_kv, c});
#endif
        }
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);   // G::load may stage through registers + ds_write
        __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
#endif

#ifdef DEBUG_V
        if (j == 0) {   // debug: V tile 0 chunk c=warp via LDS -> row-layout registers -> Og[b, 0:64, head, 128*warp : +128]
            if (warpid() < NC) {
                kv_tile<bf16> v_dbg;
                load(v_dbg, v_smem[warpid()]);
                __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
                kv_tile<float> v_dbg_f;
                copy(v_dbg_f, v_dbg);
                store<1>(g.Og, v_dbg_f, {batch_idx, 0, head_idx, warpid()});
            }
            return;
        }
#endif
#ifdef DEBUG_K
        if (j == 0) {   // debug: K tile 0 chunk c=warp, via LDS -> registers -> Og[b, 0:64, head, 128*warp : +128]
            if (warpid() < NC) {
                kv_tile<bf16> k_dbg;
                load(k_dbg, k_smem[warpid()]);
                __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
                kv_tile<float> k_dbg_f;
                copy(k_dbg_f, k_dbg);
                store<1>(g.Og, k_dbg_f, {batch_idx, 0, head_idx, warpid()});
            }
            return;
        }
#endif
        if (!warp_active || j >= my_kv_tiles) continue;   // causal: this warp is done with tile j

        // S^T = K_j Q^T over 4 head_dim chunks
        zero(att);
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            kv_tile<bf16> k_reg;
            load(k_reg, k_smem[cur + c]);
            __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
            kv_tile_transposed<bf16> k_t;
            transpose(k_t, k_reg);
#ifdef Q_STREAM   // Q chunk re-read from global (L1/L2) per KV tile: frees 128 VGPRs, scale applied to S below
            qo_tile<bf16> q_bf;
            load<1, qo_tile<bf16>, _gl_QKVO>(q_bf, g.Qg, {batch_idx, tile_idx, head_idx, c});
            __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
            qo_tile_transposed<bf16> q_tc;
            transpose(q_tc, q_bf);
            mma_AtB(att, k_t, q_tc, att);
#else
#ifndef DEBUG_NOQK_MMA   // timing probe: K read from LDS and transposed, no QK MFMA
            mma_AtB(att, k_t, q_t[c], att);
#endif
#endif
        }
#ifdef Q_STREAM
        mul(att, att, ATTN_SCALE * 1.44269504089f);
#endif
#ifdef DEBUG_S
        if (j == 0) {   // debug: S (32 q x 64 kv) of KV tile 0, row layout -> Og[b, 32*tile : +32, head, 0:64]
            rt<float, Q_BLOCK_SIZE, KV_BLOCK_SIZE, row_l, rt_32x32_s> s_row;
            transpose(s_row, att);
            store<1>(g.Og, s_row, {batch_idx, tile_idx, head_idx, 0});
            return;
        }
#endif
#ifndef DEBUG_NOMASK
        if ((j + 1) * KV_BLOCK_SIZE > q_pos0) mask_kv_tile(att, q_pos0, j, neg_inf_v, lane);
#endif

#ifdef DEBUG_PV
        // debug: P = 1 on KV tile 0 only, no softmax -> O^T = V_0^T . 1, i.e. every query gets sum over kv rows 0..63
        if (j == 0) {
            zero(att); add(att, att, 1.0f);
            copy(att_bf16, att);
            att_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_bf16);
            #pragma unroll
            for (int c = 0; c < NC; c++) {
                kv_tile<bf16, col_l, rt_16x32_4_s> v_reg;
                load(v_reg, v_smem[cur + c]);
                __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
                mma_AtB(o[c], v_reg, att_in, o[c]);
                qo_tile<float, row_l, rt_32x32_s> o_row;
                transpose(o_row, o[c]);
                store<1>(g.Og, o_row, {batch_idx, tile_idx, head_idx, c});
            }
            return;
        }
#endif
        // online softmax (per query = per column of S^T)
        col_max(max_vec, att, max_prev);
#ifdef LAZY_RESCALE
        // Lazy rescale (FlashAttention-4 style): keep the stale running max unless some query's max grew by more than
        // 2^RESCALE_LOG2 (log2 domain). P = exp2(S - stale max) is then at most 2^RESCALE_LOG2, safe in bf16 / fp32, and
        // the result stays exact because O and the row sum share the same reference max. The decision is warp-uniform,
        // so most KV tiles skip the 256-register rescale of O entirely.
#ifndef RESCALE_LOG2
#define RESCALE_LOG2 8.0f
#endif
        bool grow = false;
        {
            const float* mv = reinterpret_cast<const float*>(&max_vec.data[0][0]);
            const float* mp = reinterpret_cast<const float*>(&max_prev.data[0][0]);
            #pragma unroll
            for (int e = 0; e < (int)(sizeof(max_vec.data) / sizeof(float)); e++) grow |= (mv[e] - mp[e]) > RESCALE_LOG2;
        }
        const bool rescale = __builtin_amdgcn_ballot_w64(grow) != 0;
        if (rescale) {
            sub(scale_vec, max_prev, max_vec);
            exp2(scale_vec, scale_vec);
            mul(norm_vec, norm_vec, scale_vec);
            copy(max_prev, max_vec);
            #pragma unroll
            for (int c = 0; c < NC; c++) mul_col(o[c], o[c], scale_vec);
        }
        sub_col(att, att, max_prev);
#else
        sub(scale_vec, max_prev, max_vec);
        exp2(scale_vec, scale_vec);
        sub_col(att, att, max_vec);
        mul(norm_vec, norm_vec, scale_vec);
        copy(max_prev, max_vec);
#endif
        #pragma unroll
        for (int ti = 0; ti < att.height; ti++)
            #pragma unroll
            for (int tj = 0; tj < att.width; tj++) exp2(att.tiles[ti][tj], att.tiles[ti][tj]);
        col_sum(norm_vec, att, norm_vec);
        copy(att_bf16, att);
        att_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_bf16);

        // O_c^T = O_c^T * scale + V_c^T P^T
        #pragma unroll
        for (int c = 0; c < NC; c++) {
#if !defined(DEBUG_NORESCALE) && !defined(LAZY_RESCALE)
            mul_col(o[c], o[c], scale_vec);
#endif
#ifndef DEBUG_NOPV   // timing probe: no V LDS read, no PV MFMA
            kv_tile<bf16, col_l, rt_16x32_4_s> v_reg;
            load(v_reg, v_smem[cur + c]);
            __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
#ifndef DEBUG_NOPV_MMA   // timing probe: V read from LDS, no PV MFMA
            mma_AtB(o[c], v_reg, att_in, o[c]);
#endif
#endif
        }
    }

    if (!warp_active) return;
    #pragma unroll
    for (int c = 0; c < NC; c++) {
        div_col(o[c], o[c], norm_vec);
        qo_tile<float, row_l, rt_32x32_s> o_row;
        transpose(o_row, o[c]);
        store<1>(g.Og, o_row, {batch_idx, tile_idx, head_idx, c});
    }
    // LSE in natural log: (max * ln2) + ln(sum)
    mul(max_prev, max_prev, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_prev);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

void dispatch_d512(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker_d512, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker_d512<<<g.grid(), g.block(), mem_size, reinterpret_cast<hipStream_t>(g.stream_ptr)>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "HipKittens GQA causal attention, head_dim 512 (v0)";
    // dispatch_d512(q (B,n_q,H,D), k (B,n_kv,H_KV,D), v, o (B,n_q,H,D), lse (B,H,1,n_q), q_off, n_q, n_kv, stream)
    py::bind_function<dispatch_d512>(m, "dispatch_d512",
        &attn_globals::Qg, &attn_globals::Kg, &attn_globals::Vg, &attn_globals::Og, &attn_globals::L_vec,
        &attn_globals::q_off, &attn_globals::n_q, &attn_globals::n_kv, &attn_globals::stream_ptr);
}
