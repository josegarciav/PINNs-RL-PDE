// The residual term of a PINN loss and its weight gradient in one call, specialised per equation.
//
// jets.cpp is a general kernel: any set of derivative streams, driven by index tables, with the
// residual, the loss and the start of the reverse pass left to PyTorch. That generality costs
// about as much again as the matrix products: the recurrences through tanh run as one loop per
// term, over arrays of megabytes, and the forward and reverse passes are separate calls.
//
// This header is the other half of a translation unit that pinnrl.native.specialize generates
// for one equation (it includes jets.cpp first). The generated code supplies three functions
// written out for the equation's own streams,
//
//     tanh_fwd(A, Hn, M)      the output streams of tanh from the pre-activation streams,
//     tanh_bwd(A, H, G, M)    their adjoints, in place in G,
//     residual(out, G, X, n)  sum of squared residuals of n points and its adjoint streams,
//
// each a single loop whose body is straight-line code (what a tracing compiler such as XLA
// produces for the same step), and the driver below runs the batch in blocks of points small
// enough that every array of a block stays in the cache from the first layer of the forward
// pass to the last of the reverse pass.

#pragma once

namespace {

template <class TanhF, class TanhB, class Residual>
double blocked_residual(const float* X, int N, int L, const int* sizes, const float* const* W,
                        const float* const* b, int S, int D, const int* cols, const int* alpha,
                        const float* const* gamma, const float* const* beta, float eps,
                        const float* params, const float* mult, float scale, float* const* gW,
                        float* const* gb, float* const* ggamma, float* const* gbeta, int B,
                        TanhF tanh_fwd, TanhB tanh_bwd, Residual residual) {
    const Tables T(S, D, alpha);
    const bool norm = gamma != nullptr;
    const int d = sizes[0], k = sizes[L];
    int widest = 0;
    for (int l = 0; l <= L; ++l) widest = sizes[l] > widest ? sizes[l] : widest;
    if (B > N) B = N;
    const long full = (long)S * B;
    static thread_local std::vector<float> work, scratch, outbuf, tmp;
    const size_t need = (size_t)pinn_jets_work_size(B, L, sizes, S, norm ? 1 : 0);
    if (work.size() < need) work.resize(need);
    if (scratch.size() < 2 * (size_t)full * widest) scratch.resize(2 * (size_t)full * widest);
    if (outbuf.size() < (size_t)full * k) outbuf.resize((size_t)full * k);
    if (tmp.size() < 2 * (size_t)widest) tmp.resize(2 * (size_t)widest);
    for (int l = 0; l < L; ++l) {
        std::memset(gW[l], 0, sizeof(float) * (size_t)sizes[l] * sizes[l + 1]);
        std::memset(gb[l], 0, sizeof(float) * (size_t)sizes[l + 1]);
        if (norm && l < L - 1) {
            std::memset(ggamma[l], 0, sizeof(float) * (size_t)sizes[l + 1]);
            std::memset(gbeta[l], 0, sizeof(float) * (size_t)sizes[l + 1]);
        }
    }
    double total = 0.0;
    for (int n0 = 0; n0 < N; n0 += B) {
        const int nb = N - n0 < B ? N - n0 : B;
        const long rows = (long)S * nb;
        float* wk = work.data();
        // ---- forward
        float* h0 = wk;
        std::memcpy(h0, X + (size_t)n0 * d, sizeof(float) * (size_t)nb * d);
        std::memset(h0 + (size_t)nb * d, 0, sizeof(float) * (size_t)(rows - nb) * d);
        for (int s = 1; s < S; ++s) {
            if (T.deg[s] != 1) continue;
            int j = 0;
            while (alpha[(size_t)s * D + j] == 0) ++j;
            float* block = h0 + (size_t)s * nb * d;
            const int col = cols[j];
            for (int r = 0; r < nb; ++r) block[(size_t)r * d + col] = 1.0f;
        }
        float* out = outbuf.data();
        for (int l = 0; l < L; ++l) {
            const int in = sizes[l], width = sizes[l + 1];
            const bool last = l == L - 1;
            const float* H = wk + hin_offset(l, rows, sizes);
            float* A = last ? out : wk + act_offset(l, L, rows, sizes);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, width, in, 1.0f, H, in,
                        W[l], in, 0.0f, A, width);
            const float* bias = b[l];
            for (int r = 0; r < nb; ++r) {
                float* row = A + (size_t)r * width;
                for (int j = 0; j < width; ++j) row[j] += bias[j];
            }
            if (last) break;
            const size_t M = (size_t)nb * width;
            if (norm) {
                float* Y = wk + norm_offset(l, L, rows, sizes);
                norm_forward(T, A, Y, nb, width, gamma[l], beta[l], eps,
                             wk + norm_state_offset(l, L, rows, sizes));
                A = Y;
            }
            tanh_inplace(A, (int)M);
            tanh_fwd(A, wk + hin_offset(l + 1, rows, sizes), M);
        }
        // ---- residual: the block's sum of squares and the adjoints of the output streams
        float* Gcur = scratch.data();
        float* Gnext = scratch.data() + (size_t)full * widest;
        total += residual(out, Gcur, X + (size_t)n0 * d, nb, params, mult);
        // ---- reverse
        for (int l = L - 1; l >= 0; --l) {
            const int in = sizes[l], width = sizes[l + 1];
            const float* H = wk + hin_offset(l, rows, sizes);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, width, in, (int)rows, 1.0f, Gcur,
                        width, H, in, 1.0f, gW[l], in);
            float* bias = gb[l];
            for (int r = 0; r < nb; ++r) {
                const float* row = Gcur + (size_t)r * width;
                for (int j = 0; j < width; ++j) bias[j] += row[j];
            }
            if (l == 0) break;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, in, width, 1.0f, Gcur,
                        width, W[l], in, 0.0f, Gnext, in);
            const size_t M = (size_t)nb * in;
            const float* A = wk + (norm ? norm_offset(l - 1, L, rows, sizes)
                                        : act_offset(l - 1, L, rows, sizes));
            tanh_bwd(A, H, Gnext, M);
            if (norm) {
                float* gg = tmp.data();
                float* gbt = tmp.data() + widest;
                norm_backward(T, wk + norm_state_offset(l - 1, L, rows, sizes), Gnext, nb, in,
                              gamma[l - 1], gg, gbt);
                for (int j = 0; j < in; ++j) {
                    ggamma[l - 1][j] += gg[j];
                    gbeta[l - 1][j] += gbt[j];
                }
            }
            float* swap = Gcur;
            Gcur = Gnext;
            Gnext = swap;
        }
    }
    for (int l = 0; l < L; ++l) {
        const size_t nw = (size_t)sizes[l] * sizes[l + 1];
        for (size_t i = 0; i < nw; ++i) gW[l][i] *= scale;
        for (int j = 0; j < sizes[l + 1]; ++j) gb[l][j] *= scale;
        if (norm && l < L - 1)
            for (int j = 0; j < sizes[l + 1]; ++j) {
                ggamma[l][j] *= scale;
                gbeta[l][j] *= scale;
            }
    }
    return total * scale;
}

}  // namespace
