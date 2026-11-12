// GQA causal attention forward for head_dim 512 (Gemma 4 global layers) on CDNA4 (gfx950): split-head_dim variant.
// Two waves per SIMD. O at 32 queries x 512 is 256 fp32 registers, so each wave of a pair owns half of head_dim:
// it computes a partial S^T over its 256 dims, the pair swaps partials through LDS, both run the same softmax, and
// each does P.V on its own V half. One wave's LDS reads and softmax can then run under the other's MFMAs.
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

// 8 warps: 4 query tiles x 2 head_dim halves. Warp w and w + 4 share a SIMD and a query tile; each owns half of D.
#define NUM_WARPS 8
#define QTILES 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<QTILES>;   // K/V loads are issued by warps 0-3 only, laid out exactly as the 4-warp kernel

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


#ifndef DC_SIZE_CHECK
static_assert(DC == 64, "split kernel is written for 64-wide chunks");
#endif
constexpr int NCH = NC / 2;   // head_dim chunks per warp

struct attn_globals {
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    int q_off, n_q, n_kv;
    uint64_t stream_ptr;
    dim3 grid() { return dim3(ATTN_H, ((n_q + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE + QTILES - 1) / QTILES, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 163840; }   // the full 160 KiB of gfx950 LDS
};

#define FENCE_LGKM() do { __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt lgkmcnt(0)"); __builtin_amdgcn_sched_barrier(0); } while (0)
#define BARRIER() do { __builtin_amdgcn_sched_barrier(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0); } while (0)

__launch_bounds__(NUM_THREADS, 1)
__global__ void attend_ker_d512(const attn_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    constexpr int NBUF = 2;
    st_bf<KV_BLOCK_SIZE, DC, st_32x32_s> (&k_smem)[NBUF * NC] = al.allocate<st_bf<KV_BLOCK_SIZE, DC, st_32x32_s>, NBUF * NC>();
    st_bf<KV_BLOCK_SIZE, DC, st_8x32_s> (&v_smem)[NBUF * NC] = al.allocate<st_bf<KV_BLOCK_SIZE, DC, st_8x32_s>, NBUF * NC>();
    using att_t = attn_tile<float, col_l, rt_32x32_s>;
    constexpr int ATT_F = sizeof(att_t) / sizeof(float);   // fp32 values of one S^T tile per lane
    static_assert(ATT_F % 4 == 0, "swap uses 128-bit LDS accesses");
    float (&s_swap_raw)[NUM_WARPS][kittens::WARP_THREADS][ATT_F] = al.allocate<float, NUM_WARPS, kittens::WARP_THREADS, ATT_F>();
    typedef float lds_f4 __attribute__((ext_vector_type(4)));
    // per-lane contiguous, 128-bit LDS accesses, explicit LDS address space so the compiler emits ds_write/ds_read_b128
    auto swap_ptr = [&](int ww) {
        return (__attribute__((address_space(3))) lds_f4*)(uintptr_t)(reinterpret_cast<uintptr_t>(&s_swap_raw[ww][laneid()][0]));
    };

    const int w = warpid();
    const int qt = w % QTILES, half = w / QTILES;
    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * QTILES + qt;
    const int lane = laneid();
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;
    const bool warp_active = q_start_pos < g.n_q;
    const int q_pos0 = g.q_off + q_start_pos;
    const uint32_t neg_inf_v = 0xff800000;
    const bool loader = w < QTILES;

    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
    i32x4 k_srsrc = make_srsrc(k_base, k_row_stride * g.n_kv, k_row_stride);
    i32x4 v_srsrc = make_srsrc(v_base, v_row_stride * g.n_kv, v_row_stride);
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds[NBUF * NC], v_lds[NBUF * NC];
    #pragma unroll
    for (int c = 0; c < NBUF * NC; c++) {
        k_lds[c] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&k_smem[c].data[0]) + qt * elem_per_warp * sizeof(bf16)));
        v_lds[c] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&v_smem[c].data[0]) + qt * elem_per_warp * sizeof(bf16)));
    }
    using T = typename st_bf<KV_BLOCK_SIZE, DC, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * DC * sizeof(T) / (bytes_per_thread * QTILES * kittens::WARP_THREADS);
    static_assert(memcpy_per_tile >= 1, "tile too small for the 4 loader warps");
    uint32_t swz_K[memcpy_per_tile], swz_V[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swz_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swz_V);

    const int last_q = g.q_off + min((block_tile_idx + 1) * QTILES * Q_BLOCK_SIZE, g.n_q);
    const int num_kv_tiles = min((last_q + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE, (g.n_kv + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE);
    const int my_kv_tiles = (q_pos0 + Q_BLOCK_SIZE + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    // Q: this warp's NCH chunks, row-major, pre-scaled by softmax_scale * log2(e)
    qo_tile<bf16> q_r[NCH];
    if (warp_active) {
        #pragma unroll
        for (int cc = 0; cc < NCH; cc++) {
            qo_tile<float> q_fl;
            load<1, qo_tile<float>, _gl_QKVO>(q_fl, g.Qg, {batch_idx, tile_idx, head_idx, half * NCH + cc});
            mul(q_fl, q_fl, ATTN_SCALE * 1.44269504089f);
            copy(q_r[cc], q_fl);
        }
    }
    qo_tile_transposed<float, col_l, rt_32x32_s> o[NCH];
    #pragma unroll
    for (int cc = 0; cc < NCH; cc++) zero(o[cc]);
    att_t att;
    attn_tile<bf16, col_l, rt_32x32_s> att_bf16;
    attn_tile<bf16, col_l, rt_16x32_4_s> att_in;
    typename att_t::row_vec max_vec, max_prev, norm_vec, scale_vec;
    zero(max_prev);
    add(max_prev, max_prev, -1.0e30f);
    zero(norm_vec);

    constexpr int LOADS_PER_TILE = 2 * NC * memcpy_per_tile;
    static_assert(LOADS_PER_TILE <= 63, "vmcnt field is 6 bits");
    if (loader) {
        #pragma unroll
        for (int c = 0; c < NC; c++) {
            G::load<1, false>(k_smem[c], g.Kg, {batch_idx, 0, head_idx_kv, c}, swz_K, k_srsrc, k_base, k_lds[c]);
            G::load<1, false>(v_smem[c], g.Vg, {batch_idx, 0, head_idx_kv, c}, swz_V, v_srsrc, v_base, v_lds[c]);
        }
    }
    for (int j = 0; j < num_kv_tiles; j++) {
        const int cur = (j & 1) * NC;
        BARRIER();   // everyone is done with tile j-1 (its K/V buffer and the S swap slots)
        if (loader) {
            if (j + 1 < num_kv_tiles) {
                const int nxt = ((j + 1) & 1) * NC;
                #pragma unroll
                for (int c = 0; c < NC; c++) {
                    G::load<1, false>(k_smem[nxt + c], g.Kg, {batch_idx, j + 1, head_idx_kv, c}, swz_K, k_srsrc, k_base, k_lds[nxt + c]);
                    G::load<1, false>(v_smem[nxt + c], g.Vg, {batch_idx, j + 1, head_idx_kv, c}, swz_V, v_srsrc, v_base, v_lds[nxt + c]);
                }
                __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(%0)" :: "n"(LOADS_PER_TILE)); __builtin_amdgcn_sched_barrier(0);
            } else {
                __builtin_amdgcn_sched_barrier(0); asm volatile("s_waitcnt vmcnt(0)"); __builtin_amdgcn_sched_barrier(0);
            }
        }
        BARRIER();   // tile j is in LDS
        const bool act = warp_active && j < my_kv_tiles;

        // partial S^T over this warp's half of head_dim
        if (act) {
            zero(att);
            #pragma unroll
            for (int cc = 0; cc < NCH; cc++) {
                kv_tile<bf16> k_reg;
                load(k_reg, k_smem[cur + half * NCH + cc]);
                FENCE_LGKM();
                mma_ABt(att, k_reg, q_r[cc], att);
            }
#ifndef DEBUG_NOSWAP
            const lds_f4* af = reinterpret_cast<const lds_f4*>(&att);
            auto dst = swap_ptr(w);
            #pragma unroll
            for (int e = 0; e < ATT_F / 4; e++) dst[e] = af[e];
#endif
        }
#ifndef DEBUG_NOSWAP
        FENCE_LGKM();
        BARRIER();   // partials visible to the partner
#endif
        if (!act) continue;
#ifndef DEBUG_NOSWAP
        {
            lds_f4* af = reinterpret_cast<lds_f4*>(&att);
            auto src = swap_ptr(w ^ QTILES);
            #pragma unroll
            for (int e = 0; e < ATT_F / 4; e++) af[e] += src[e];   // a + b == b + a: both waves get identical S
        }
#endif
        if ((j + 1) * KV_BLOCK_SIZE > q_pos0) mask_kv_tile(att, q_pos0, j, neg_inf_v, lane);

        // online softmax with lazy rescale (identical in both waves of the pair)
        col_max(max_vec, att, max_prev);
        bool grow = false;
        {
            const float* mv = reinterpret_cast<const float*>(&max_vec.data[0][0]);
            const float* mp = reinterpret_cast<const float*>(&max_prev.data[0][0]);
            #pragma unroll
            for (int e = 0; e < (int)(sizeof(max_vec.data) / sizeof(float)); e++) grow |= (mv[e] - mp[e]) > 8.0f;
        }
        if (__builtin_amdgcn_ballot_w64(grow) != 0) {
            sub(scale_vec, max_prev, max_vec);
            exp2(scale_vec, scale_vec);
            mul(norm_vec, norm_vec, scale_vec);
            copy(max_prev, max_vec);
            #pragma unroll
            for (int cc = 0; cc < NCH; cc++) mul_col(o[cc], o[cc], scale_vec);
        }
        sub_col(att, att, max_prev);
        #pragma unroll
        for (int ti = 0; ti < att.height; ti++)
            #pragma unroll
            for (int tj = 0; tj < att.width; tj++) exp2(att.tiles[ti][tj], att.tiles[ti][tj]);
        col_sum(norm_vec, att, norm_vec);
        copy(att_bf16, att);
        att_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_bf16);

        #pragma unroll
        for (int cc = 0; cc < NCH; cc++) {
            kv_tile<bf16, col_l, rt_16x32_4_s> v_reg;
            load(v_reg, v_smem[cur + half * NCH + cc]);
            FENCE_LGKM();
            mma_AtB(o[cc], v_reg, att_in, o[cc]);
        }
    }

    if (!warp_active) return;
    #pragma unroll
    for (int cc = 0; cc < NCH; cc++) {
        div_col(o[cc], o[cc], norm_vec);
        qo_tile<float, row_l, rt_32x32_s> o_row;
        transpose(o_row, o[cc]);
        store<1>(g.Og, o_row, {batch_idx, tile_idx, head_idx, half * NCH + cc});
    }
    if (half == 0) {
        mul(max_prev, max_prev, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_prev);
        store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
    }
}

void dispatch_d512(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker_d512, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker_d512<<<g.grid(), g.block(), mem_size, reinterpret_cast<hipStream_t>(g.stream_ptr)>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "HipKittens GQA causal attention, head_dim 512, split-head_dim (2 waves per SIMD)";
    py::bind_function<dispatch_d512>(m, "dispatch_d512",
        &attn_globals::Qg, &attn_globals::Kg, &attn_globals::Vg, &attn_globals::Og, &attn_globals::L_vec,
        &attn_globals::q_off, &attn_globals::n_q, &attn_globals::n_kv, &attn_globals::stream_ptr);
}
