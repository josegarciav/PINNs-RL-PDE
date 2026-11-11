// Forward derivatives ("jets") of a tanh MLP and the reverse pass for its weights.
//
// For a network u(x) with layers a = W h + b, h = tanh(a), this kernel carries, next to the
// value, the first derivatives with respect to chosen inputs and the pure second derivatives
// with respect to some of those, layer by layer (Taylor mode):
//
//     h   = s            = tanh(a)
//     h_i = d1 a_i                      d1 = 1 - s^2
//     h_ii = d2 a_i^2 + d1 a_ii         d2 = -2 s d1
//
// and, given the adjoints of the outputs, the gradient with respect to every weight and bias in
// one hand-written reverse pass (d3 = -2 d1 (1 - 3 s^2)). All streams of a batch of N points are
// stacked as row blocks of one (S N x width) matrix, so a layer is one sgemm; the elementwise
// work is fused into loops the compiler vectorises. Experiment I11 measures it against autograd.
//
// With layer normalization (Linear, LayerNorm, tanh, the packaged feed-forward default) each
// row of the pre-activation is normalised over its features, y = gamma (a - mean) / sigma + beta,
// and the derivative streams follow by the quotient rule to second order; see norm_forward and
// norm_backward below.
//
// Streams: 0 is the value; 1..nf the first derivatives (input column first[i]); nf+1..nf+ns the
// second derivatives, parent[j] giving the index (0..nf-1) of the first derivative they extend.
// Float32, row-major; W[l] is (out x in) as in torch.nn.Linear.

#include <Accelerate/Accelerate.h>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

const int CHUNK = 1024;  // elements handled together: 4 kB an array, well inside the L1 cache

inline size_t hin_offset(int l, long rows, const int* sizes) {
    size_t off = 0;
    for (int k = 0; k < l; ++k) off += (size_t)rows * sizes[k];
    return off;
}

inline size_t act_offset(int l, int L, long rows, const int* sizes) {
    size_t off = hin_offset(L, rows, sizes);
    for (int k = 0; k < l; ++k) off += (size_t)rows * sizes[k + 1];
    return off;
}

// The post-normalization streams of hidden layer l (only with layer normalization).
inline size_t norm_offset(int l, int L, long rows, const int* sizes) {
    size_t off = act_offset(L - 1, L, rows, sizes);
    for (int k = 0; k < l; ++k) off += (size_t)rows * sizes[k + 1];
    return off;
}

// Layer normalization of one row and its derivative streams. With c = a - mean(a),
// v = mean(c^2) + eps, r = 1/sqrt(v), z = c r:
//   first derivative i:   c_i = a_i - mean(a_i),  s_i = mean(c c_i) r,  z_i = (c_i - z s_i) r
//   second derivative j (of first i):  c_jj = a_jj - mean(a_jj),  q = mean(c_i^2 + c c_jj),
//                         s_jj = (q - s_i^2) r,  z_jj = (c_jj - 2 z_i s_i - z s_jj) r
struct NormRow {
    int w, nf, ns;
    std::vector<float> c, z, ci, zi, cjj;  // c, z: w; ci, zi: nf*w; cjj: ns*w
    std::vector<float> si, sjj, q;         // per stream scalars
    float r = 0.0f;
    NormRow(int w_, int nf_, int ns_)
        : w(w_), nf(nf_), ns(ns_), c(w_), z(w_), ci((size_t)nf_ * w_), zi((size_t)nf_ * w_),
          cjj((size_t)ns_ * w_), si(nf_), sjj(ns_), q(ns_) {}

    // a: the row's value stream; step: distance between streams of the same row (N * w floats).
    void compute(const float* a, size_t step, const int* parent, float eps) {
        const float inv = 1.0f / w;
        float mean = 0.0f;
        for (int k = 0; k < w; ++k) mean += a[k];
        mean *= inv;
        float v = 0.0f;
        for (int k = 0; k < w; ++k) {
            c[k] = a[k] - mean;
            v += c[k] * c[k];
        }
        r = 1.0f / std::sqrt(v * inv + eps);
        for (int k = 0; k < w; ++k) z[k] = c[k] * r;
        for (int i = 0; i < nf; ++i) {
            const float* ai = a + (size_t)(1 + i) * step;
            float* cc = ci.data() + (size_t)i * w;
            float* zz = zi.data() + (size_t)i * w;
            float m = 0.0f;
            for (int k = 0; k < w; ++k) m += ai[k];
            m *= inv;
            float pi = 0.0f;
            for (int k = 0; k < w; ++k) {
                cc[k] = ai[k] - m;
                pi += c[k] * cc[k];
            }
            si[i] = pi * inv * r;
            for (int k = 0; k < w; ++k) zz[k] = (cc[k] - z[k] * si[i]) * r;
        }
    }

    void compute_second(const float* a, size_t step, const int* parent) {
        const float inv = 1.0f / w;
        for (int j = 0; j < ns; ++j) {
            const int i = parent[j];
            const float* ajj = a + (size_t)(1 + nf + j) * step;
            const float* cc = ci.data() + (size_t)i * w;
            float* cj = cjj.data() + (size_t)j * w;
            float m = 0.0f;
            for (int k = 0; k < w; ++k) m += ajj[k];
            m *= inv;
            float acc = 0.0f;
            for (int k = 0; k < w; ++k) {
                cj[k] = ajj[k] - m;
                acc += cc[k] * cc[k] + c[k] * cj[k];
            }
            q[j] = acc * inv;
            sjj[j] = (q[j] - si[i] * si[i]) * r;
        }
    }
};

// y streams (N rows) = gamma z + beta from the a streams; A and Y are (S N x w).
void norm_forward(const float* A, float* Y, int N, int w, int nf, int ns, const int* parent,
                  const float* gamma, const float* beta, float eps) {
    NormRow row(w, nf, ns);
    const size_t step = (size_t)N * w;
    for (int n = 0; n < N; ++n) {
        const float* a = A + (size_t)n * w;
        float* y = Y + (size_t)n * w;
        row.compute(a, step, parent, eps);
        row.compute_second(a, step, parent);
        for (int k = 0; k < w; ++k) y[k] = gamma[k] * row.z[k] + beta[k];
        for (int i = 0; i < nf; ++i) {
            float* yi = y + (size_t)(1 + i) * step;
            const float* zz = row.zi.data() + (size_t)i * w;
            for (int k = 0; k < w; ++k) yi[k] = gamma[k] * zz[k];
        }
        for (int j = 0; j < ns; ++j) {
            const int i = parent[j];
            float* yjj = y + (size_t)(1 + nf + j) * step;
            const float* zz = row.zi.data() + (size_t)i * w;
            const float* cj = row.cjj.data() + (size_t)j * w;
            const float si = row.si[i], sjj = row.sjj[j];
            for (int k = 0; k < w; ++k)
                yjj[k] = gamma[k] * (cj[k] - 2.0f * zz[k] * si - row.z[k] * sjj) * row.r;
        }
    }
}

// G holds the adjoints of the y streams on entry and of the a streams on exit;
// ggamma and gbeta (w each) are overwritten with the gradients of gamma and beta.
void norm_backward(const float* A, float* G, int N, int w, int nf, int ns, const int* parent,
                   const float* gamma, float eps, float* ggamma, float* gbeta) {
    NormRow row(w, nf, ns);
    const size_t step = (size_t)N * w;
    const float inv = 1.0f / w;
    std::vector<float> gz(w), gc(w), gT(w), gzi((size_t)nf * w), gci((size_t)nf * w);
    std::vector<float> gsi(nf);
    for (int k = 0; k < w; ++k) ggamma[k] = gbeta[k] = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* a = A + (size_t)n * w;
        float* g = G + (size_t)n * w;
        row.compute(a, step, parent, eps);
        row.compute_second(a, step, parent);
        const float r = row.r;
        float gr = 0.0f;
        for (int k = 0; k < w; ++k) {
            ggamma[k] += g[k] * row.z[k];
            gbeta[k] += g[k];
            gz[k] = gamma[k] * g[k];
            gc[k] = 0.0f;
        }
        for (int i = 0; i < nf; ++i) {
            const float* gi = g + (size_t)(1 + i) * step;
            const float* zz = row.zi.data() + (size_t)i * w;
            float* gzz = gzi.data() + (size_t)i * w;
            float* gcc = gci.data() + (size_t)i * w;
            for (int k = 0; k < w; ++k) {
                ggamma[k] += gi[k] * zz[k];
                gzz[k] = gamma[k] * gi[k];
                gcc[k] = 0.0f;
            }
            gsi[i] = 0.0f;
        }
        for (int j = 0; j < ns; ++j) {
            const int i = parent[j];
            float* gjj = g + (size_t)(1 + nf + j) * step;
            const float* zz = row.zi.data() + (size_t)i * w;
            const float* cc = row.ci.data() + (size_t)i * w;
            const float* cj = row.cjj.data() + (size_t)j * w;
            float* gzz = gzi.data() + (size_t)i * w;
            float* gcc = gci.data() + (size_t)i * w;
            const float si = row.si[i], sjj = row.sjj[j];
            float gsjj = 0.0f, gsi_acc = 0.0f;
            for (int k = 0; k < w; ++k) {
                const float T = cj[k] - 2.0f * zz[k] * si - row.z[k] * sjj;  // z_jj = T r
                ggamma[k] += gjj[k] * T * r;
                const float gzjj = gamma[k] * gjj[k];
                gr += gzjj * T;
                gT[k] = gzjj * r;
                gzz[k] += -2.0f * si * gT[k];
                gsi_acc += gT[k] * zz[k];
                gz[k] += -sjj * gT[k];
                gsjj -= gT[k] * row.z[k];
            }
            gsi[i] += -2.0f * gsi_acc - 2.0f * si * r * gsjj;
            gr += gsjj * (row.q[j] - si * si);
            const float gq = gsjj * r * inv;
            float mean = 0.0f;
            for (int k = 0; k < w; ++k) {
                gcc[k] += gq * 2.0f * cc[k];
                gc[k] += gq * cj[k];
                gT[k] += gq * row.c[k];  // gT is now the adjoint of c_jj
                mean += gT[k];
            }
            mean *= inv;
            for (int k = 0; k < w; ++k) gjj[k] = gT[k] - mean;
        }
        for (int i = 0; i < nf; ++i) {
            float* gi = g + (size_t)(1 + i) * step;
            const float* zz = row.zi.data() + (size_t)i * w;
            const float* cc = row.ci.data() + (size_t)i * w;
            const float* gzz = gzi.data() + (size_t)i * w;
            float* gcc = gci.data() + (size_t)i * w;
            const float si = row.si[i];
            float gs = gsi[i];
            for (int k = 0; k < w; ++k) {
                gr += gzz[k] * zz[k] / r;  // U = z_i / r
                const float gU = gzz[k] * r;
                gcc[k] += gU;
                gz[k] += -si * gU;
                gs -= gU * row.z[k];
            }
            gr += gs * si / r;  // p_i = s_i / r
            const float gp = gs * r * inv;
            float mean = 0.0f;
            for (int k = 0; k < w; ++k) {
                gc[k] += gp * cc[k];
                gcc[k] += gp * row.c[k];
                mean += gcc[k];
            }
            mean *= inv;
            for (int k = 0; k < w; ++k) gi[k] = gcc[k] - mean;
        }
        for (int k = 0; k < w; ++k) {
            gc[k] += gz[k] * r;
            gr += gz[k] * row.c[k];
        }
        const float gv = -0.5f * gr * r * r * r;
        float mean = 0.0f;
        for (int k = 0; k < w; ++k) {
            gc[k] += gv * 2.0f * row.c[k] * inv;
            mean += gc[k];
        }
        mean *= inv;
        for (int k = 0; k < w; ++k) g[k] = gc[k] - mean;
    }
}

}  // namespace

extern "C" {

// Floats of work storage the forward pass fills and the backward pass reads.
long pinn_jets_work_size(int N, int L, const int* sizes, int S, int with_norm) {
    const long rows = (long)S * N;
    return (long)(with_norm ? norm_offset(L - 1, L, rows, sizes) : act_offset(L - 1, L, rows, sizes));
}

// out: (S N x sizes[L]) row blocks per stream.
// gamma, beta: per hidden layer the LayerNorm weight and bias, or null without normalization.
void pinn_jets_forward(const float* X, int N, int L, const int* sizes, const float* const* W,
                       const float* const* b, int nf, const int* first, int ns, const int* parent,
                       const float* const* gamma, const float* const* beta, float eps, float* work,
                       float* out) {
    const int S = 1 + nf + ns;
    const long rows = (long)S * N;
    const int d = sizes[0];
    float* h0 = work;
    std::memcpy(h0, X, sizeof(float) * (size_t)N * d);
    std::memset(h0 + (size_t)N * d, 0, sizeof(float) * (size_t)(rows - N) * d);
    for (int i = 0; i < nf; ++i) {
        float* block = h0 + (size_t)(1 + i) * N * d;
        const int col = first[i];
        for (int r = 0; r < N; ++r) block[(size_t)r * d + col] = 1.0f;
    }
    for (int l = 0; l < L; ++l) {
        const int in = sizes[l], width = sizes[l + 1];
        const bool last = l == L - 1;
        const float* H = work + hin_offset(l, rows, sizes);
        float* A = last ? out : work + act_offset(l, L, rows, sizes);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, width, in, 1.0f, H, in, W[l], in,
                    0.0f, A, width);
        const float* bias = b[l];
        for (int r = 0; r < N; ++r) {
            float* row = A + (size_t)r * width;
            for (int j = 0; j < width; ++j) row[j] += bias[j];
        }
        if (last) break;
        const size_t M = (size_t)N * width;
        const int count = (int)M;
        if (gamma != nullptr) {  // normalise; the activation then reads the y streams
            float* Y = work + norm_offset(l, L, rows, sizes);
            norm_forward(A, Y, N, width, nf, ns, parent, gamma[l], beta[l], eps);
            A = Y;
        }
        vvtanhf(A, A, &count);
        float* Hn = work + hin_offset(l + 1, rows, sizes);
        // The streams are processed a chunk at a time, so tanh's derivatives are computed once
        // and every stream's loop finds its operands in the first-level cache.
        float d1[CHUNK], d2[CHUNK];
        for (size_t start = 0; start < M; start += CHUNK) {
            const int len = (int)((M - start) < (size_t)CHUNK ? (M - start) : (size_t)CHUNK);
            const float* s = A + start;
            float* h = Hn + start;
            for (int q = 0; q < len; ++q) {
                h[q] = s[q];
                d1[q] = 1.0f - s[q] * s[q];
                d2[q] = -2.0f * s[q] * d1[q];
            }
            for (int i = 0; i < nf; ++i) {
                const float* ax = A + (size_t)(1 + i) * M + start;
                float* hx = Hn + (size_t)(1 + i) * M + start;
                for (int q = 0; q < len; ++q) hx[q] = d1[q] * ax[q];
            }
            for (int j = 0; j < ns; ++j) {
                const float* ax = A + (size_t)(1 + parent[j]) * M + start;
                const float* axx = A + (size_t)(1 + nf + j) * M + start;
                float* hxx = Hn + (size_t)(1 + nf + j) * M + start;
                for (int q = 0; q < len; ++q) hxx[q] = d2[q] * ax[q] * ax[q] + d1[q] * axx[q];
            }
        }
    }
}

// g_out: adjoints of `out`. gW[l] (out x in) and gb[l] are overwritten with the gradients.
// scratch: 2 * S * N * (largest layer width) floats.
// ggamma[l], gbeta[l]: gradients of the LayerNorm parameters (unused without normalization).
void pinn_jets_backward(int N, int L, const int* sizes, const float* const* W, int nf, int ns,
                        const int* parent, const float* const* gamma, float eps, const float* work,
                        const float* g_out, float* const* gW, float* const* gb, float* const* ggamma,
                        float* const* gbeta, float* scratch) {
    const int S = 1 + nf + ns;
    const long rows = (long)S * N;
    int widest = 0;
    for (int l = 0; l <= L; ++l) widest = sizes[l] > widest ? sizes[l] : widest;
    float* Gcur = scratch;
    float* Gnext = scratch + (size_t)rows * widest;
    std::memcpy(Gcur, g_out, sizeof(float) * (size_t)rows * sizes[L]);
    for (int l = L - 1; l >= 0; --l) {
        const int in = sizes[l], width = sizes[l + 1];
        const float* H = work + hin_offset(l, rows, sizes);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, width, in, (int)rows, 1.0f, Gcur, width, H, in,
                    0.0f, gW[l], in);
        float* bias = gb[l];
        for (int j = 0; j < width; ++j) bias[j] = 0.0f;
        for (int r = 0; r < N; ++r) {
            const float* row = Gcur + (size_t)r * width;
            for (int j = 0; j < width; ++j) bias[j] += row[j];
        }
        if (l == 0) break;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, in, width, 1.0f, Gcur, width, W[l],
                    in, 0.0f, Gnext, in);
        const size_t M = (size_t)N * in;
        const float* A = work + (gamma != nullptr ? norm_offset(l - 1, L, rows, sizes)
                                                  : act_offset(l - 1, L, rows, sizes));
        float d1[CHUNK], d2[CHUNK], d3[CHUNK];
        for (size_t start = 0; start < M; start += CHUNK) {
            const int len = (int)((M - start) < (size_t)CHUNK ? (M - start) : (size_t)CHUNK);
            const float* s = A + start;
            float* gh = Gnext + start;
            for (int q = 0; q < len; ++q) {
                d1[q] = 1.0f - s[q] * s[q];
                d2[q] = -2.0f * s[q] * d1[q];
                d3[q] = -2.0f * d1[q] * (1.0f - 3.0f * s[q] * s[q]);
                gh[q] *= d1[q];
            }
            // the value's adjoint collects from every stream, read before they are overwritten
            for (int i = 0; i < nf; ++i) {
                const float* ax = A + (size_t)(1 + i) * M + start;
                const float* ghx = Gnext + (size_t)(1 + i) * M + start;
                for (int q = 0; q < len; ++q) gh[q] += ghx[q] * d2[q] * ax[q];
            }
            for (int j = 0; j < ns; ++j) {
                const float* ax = A + (size_t)(1 + parent[j]) * M + start;
                const float* axx = A + (size_t)(1 + nf + j) * M + start;
                const float* ghxx = Gnext + (size_t)(1 + nf + j) * M + start;
                for (int q = 0; q < len; ++q) gh[q] += ghxx[q] * (d3[q] * ax[q] * ax[q] + d2[q] * axx[q]);
            }
            for (int i = 0; i < nf; ++i) {
                const float* ax = A + (size_t)(1 + i) * M + start;
                float* ghx = Gnext + (size_t)(1 + i) * M + start;
                for (int q = 0; q < len; ++q) ghx[q] *= d1[q];
                for (int j = 0; j < ns; ++j) {
                    if (parent[j] != i) continue;
                    const float* ghxx = Gnext + (size_t)(1 + nf + j) * M + start;
                    for (int q = 0; q < len; ++q) ghx[q] += ghxx[q] * 2.0f * d2[q] * ax[q];
                }
            }
            for (int j = 0; j < ns; ++j) {
                float* ghxx = Gnext + (size_t)(1 + nf + j) * M + start;
                for (int q = 0; q < len; ++q) ghxx[q] *= d1[q];
            }
        }
        if (gamma != nullptr)  // from the adjoints of the normalised streams to the pre-norm ones
            norm_backward(work + act_offset(l - 1, L, rows, sizes), Gnext, N, in, nf, ns, parent,
                          gamma[l - 1], eps, ggamma[l - 1], gbeta[l - 1]);
        float* tmp = Gcur;
        Gcur = Gnext;
        Gnext = tmp;
    }
}

}  // extern "C"
