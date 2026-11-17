// Forward derivatives ("jets") of a tanh MLP and the reverse pass for its weights.
//
// For a network u(x) with layers a = W h + b, h = tanh(a), this kernel carries, next to the
// value, the Taylor coefficients of u with respect to chosen inputs, layer by layer (Taylor mode).
// Stream 0 is the value; stream s >= 1 is the Taylor coefficient of the multi-index alpha[s]
// over the D differentiated input columns, that is the derivative divided by alpha!. The set of
// multi-indices is downward closed (every beta <= alpha of a requested alpha is present) and
// sorted by total degree, so every recurrence below reads streams written before it.
//
// Through a linear layer the coefficients pass linearly (the bias enters the value only). Through
// tanh they follow the recurrence of s = tanh(a), s' = (1 - s^2) a' = q a': for alpha with
// alpha_i > 0,
//
//     alpha_i s_alpha = sum_{beta <= alpha, beta_i >= 1} beta_i a_beta q_{alpha - beta},
//     q_gamma = delta_{gamma 0} - sum_{beta <= gamma} s_beta s_{gamma - beta},
//
// and the reverse pass is the same recurrence read backwards. With layer normalization
// (Linear, LayerNorm, tanh, the packaged feed-forward default) each row of the pre-activation
// is normalised over its features, y = gamma (a - mean) r + beta with r = (mean(c^2) + eps)^-1/2,
// and the coefficient streams follow by Taylor arithmetic for the mean, the product and the
// inverse square root (norm_forward and norm_backward).
//
// All streams of a batch of N points are stacked as row blocks of one (S N x width) matrix, so a
// layer is one sgemm (Accelerate on macOS, OpenBLAS elsewhere); the elementwise work is fused into
// loops the compiler vectorises. Experiments I11–I14 measure it against autograd. Float32,
// row-major; W[l] is (out x in) as in torch.nn.Linear.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>  // OpenBLAS (libopenblas-dev); see the build flags in __init__.py
#include "fast_tanh.h"
#endif

namespace {

// x = tanh(x) over n elements: Apple's vectorised vvtanhf, or a branch-free tanh the compiler
// vectorises (the C library's tanhf is scalar, and glibc's vector one needs full fast-math).
inline void tanh_inplace(float* x, int n) {
#ifdef __APPLE__
    vvtanhf(x, x, &n);
#else
    for (int i = 0; i < n; ++i) x[i] = fast_math::tanhf_approx(x[i]);
#endif
}

}  // namespace

namespace {

// Elements handled together: a few kB an array, so every stream's chunk sits in the L1 cache.
inline int chunk_size(int S) { return S <= 8 ? 1024 : 512; }

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

// The normalization state of hidden layer l, saved by the forward pass for the reverse pass:
// per row the centred streams c and the normalized streams z (S x w each), then v and r (S each).
inline size_t norm_state_offset(int l, int L, long rows, const int* sizes) {
    size_t off = norm_offset(L - 1, L, rows, sizes);
    for (int k = 0; k < l; ++k) off += 2 * (size_t)rows * sizes[k + 1] + 2 * (size_t)rows;
    return off;
}

// The index tables of the recurrences, built once per call from the multi-indices.
struct Tables {
    int S, D, maxdeg;
    std::vector<int> alpha;  // S x D
    std::vector<int> deg;    // S
    // tanh: s_t = sum_terms coef a[a_idx] q[q_idx]; the term beta = alpha (coef 1, q_0) is kept
    // apart as the stream's own term, the others are in [tstart[t], tstart[t+1])
    std::vector<int> tstart, t_a, t_q;
    std::vector<float> t_coef;
    std::vector<char> q_needed;  // whether q_gamma is read by any term
    // ordered pairs (b, c) with b + c = gamma, in [pstart[g], pstart[g+1]) (the product c r)
    std::vector<int> pstart, p_b, p_c;
    // unordered pairs {b, c} with b + c = gamma and a factor 2 for b != c (symmetric products)
    std::vector<int> ustart, u_b, u_c;
    std::vector<float> u_f;
    // inverse square root: r_t = sum_terms coef r[r_idx] v[v_idx] / (v_0 rdiv[t])
    std::vector<int> rstart, r_r, r_v, rdiv;
    std::vector<float> r_coef;

    int find(const std::vector<int>& m) const {
        for (int s = 0; s < S; ++s) {
            bool same = true;
            for (int k = 0; k < D && same; ++k) same = alpha[(size_t)s * D + k] == m[k];
            if (same) return s;
        }
        return -1;
    }

    Tables(int S_, int D_, const int* alpha_) : S(S_), D(D_), maxdeg(0), alpha(alpha_, alpha_ + (size_t)S_ * D_) {
        deg.resize(S);
        for (int s = 0; s < S; ++s) {
            int d = 0;
            for (int k = 0; k < D; ++k) d += alpha[(size_t)s * D + k];
            deg[s] = d;
            maxdeg = d > maxdeg ? d : maxdeg;
        }
        std::vector<int> c(D);
        tstart.push_back(0);
        pstart.push_back(0);
        ustart.push_back(0);
        rstart.push_back(0);
        rdiv.assign(S, 1);
        q_needed.assign(S, 0);
        q_needed[0] = S > 1;
        for (int s = 0; s < S; ++s) {
            const int* al = alpha.data() + (size_t)s * D;
            int i = 0;
            while (i < D && al[i] == 0) ++i;  // the variable the recurrence differentiates along
            if (s > 0) rdiv[s] = al[i];
            for (int t = 0; t < S; ++t) {  // every beta <= alpha in the set
                const int* be = alpha.data() + (size_t)t * D;
                bool le = true;
                for (int k = 0; k < D && le; ++k) le = be[k] <= al[k];
                if (!le) continue;
                for (int k = 0; k < D; ++k) c[k] = al[k] - be[k];
                const int g = find(c);  // alpha - beta, present by downward closure
                p_b.push_back(t);
                p_c.push_back(g);
                if (t <= g) {
                    u_b.push_back(t);
                    u_c.push_back(g);
                    u_f.push_back(t == g ? 1.0f : 2.0f);
                }
                if (s > 0 && be[i] >= 1 && t != s) {
                    t_a.push_back(t);
                    t_q.push_back(g);
                    t_coef.push_back((float)be[i] / (float)al[i]);
                    q_needed[g] = 1;
                }
                if (s > 0 && c[i] >= 1) {
                    // -1/2 (D_i v) r = v (D_i r), solved for r_alpha (the beta = 0 term of the
                    // right-hand side holds the unknown)
                    r_r.push_back(t);
                    r_v.push_back(g);
                    r_coef.push_back(-0.5f * (float)c[i]);
                    if (t != 0) {
                        r_v.push_back(t);
                        r_r.push_back(g);
                        r_coef.push_back(-(float)c[i]);
                    }
                }
            }
            tstart.push_back((int)t_a.size());
            pstart.push_back((int)p_b.size());
            ustart.push_back((int)u_b.size());
            rstart.push_back((int)r_r.size());
        }
    }
};

// ------------------------------------------------------------------------------- tanh streams
// A: the pre-activation streams of a layer, stream 0 already replaced by s_0 = tanh(a_0).
// Hn: the output streams (h = s). M = N * width; the chunk is [start, start + len); q holds the
// q streams of the chunk (stride C).
void tanh_forward_chunk(const Tables& T, const float* A, float* Hn, size_t M, size_t start, int len,
                        float* q, int C) {
    const int S = T.S;
    const float* s0 = A + start;
    float* h0 = Hn + start;
    for (int e = 0; e < len; ++e) {
        h0[e] = s0[e];
        q[e] = 1.0f - s0[e] * s0[e];
    }
    for (int t = 1; t < S; ++t) {
        float* ht = Hn + (size_t)t * M + start;
        const float* at = A + (size_t)t * M + start;
        for (int e = 0; e < len; ++e) ht[e] = at[e] * q[e];  // the stream's own term, a_t q_0
        for (int k = T.tstart[t]; k < T.tstart[t + 1]; ++k) {
            const float* a = A + (size_t)T.t_a[k] * M + start;
            const float* qq = q + (size_t)T.t_q[k] * C;
            const float coef = T.t_coef[k];
            for (int e = 0; e < len; ++e) ht[e] += coef * a[e] * qq[e];
        }
        if (T.q_needed[t]) {
            float* qt = q + (size_t)t * C;
            for (int k = T.ustart[t]; k < T.ustart[t + 1]; ++k) {
                const float* hb = Hn + (size_t)T.u_b[k] * M + start;
                const float* hc = Hn + (size_t)T.u_c[k] * M + start;
                const float f = -T.u_f[k];
                if (k == T.ustart[t])
                    for (int e = 0; e < len; ++e) qt[e] = f * hb[e] * hc[e];
                else
                    for (int e = 0; e < len; ++e) qt[e] += f * hb[e] * hc[e];
            }
        }
    }
}

// G: on entry the adjoints of the h streams, on exit those of the a streams (in place).
// H: the h streams (s); A: the a streams (stream 0 holds s_0). ahat and qhat: S x C scratch.
void tanh_backward_chunk(const Tables& T, const float* A, const float* H, float* G, size_t M,
                         size_t start, int len, float* q, float* ahat, float* qhat, int C,
                         char* awritten, char* qwritten) {
    const int S = T.S;
    const float* s0 = H + start;
    for (int e = 0; e < len; ++e) q[e] = 1.0f - s0[e] * s0[e];
    for (int t = 1; t < S; ++t) {
        if (!T.q_needed[t]) continue;
        float* qt = q + (size_t)t * C;
        for (int k = T.ustart[t]; k < T.ustart[t + 1]; ++k) {
            const float* hb = H + (size_t)T.u_b[k] * M + start;
            const float* hc = H + (size_t)T.u_c[k] * M + start;
            const float f = -T.u_f[k];
            if (k == T.ustart[t])
                for (int e = 0; e < len; ++e) qt[e] = f * hb[e] * hc[e];
            else
                for (int e = 0; e < len; ++e) qt[e] += f * hb[e] * hc[e];
        }
    }
    for (int t = 0; t < S; ++t) awritten[t] = qwritten[t] = 0;
    for (int d = T.maxdeg; d >= 1; --d) {
        for (int t = 1; t < S; ++t) {
            if (T.deg[t] != d) continue;
            float* gt = G + (size_t)t * M + start;
            for (int k = T.tstart[t]; k < T.tstart[t + 1]; ++k) {
                const int ia = T.t_a[k], iq = T.t_q[k];
                const float* a = A + (size_t)ia * M + start;
                const float* qq = q + (size_t)iq * C;
                float* ah = ahat + (size_t)ia * C;
                float* qh = qhat + (size_t)iq * C;
                const float coef = T.t_coef[k];
                if (awritten[ia] && qwritten[iq]) {
                    for (int e = 0; e < len; ++e) {
                        ah[e] += coef * gt[e] * qq[e];
                        qh[e] += coef * gt[e] * a[e];
                    }
                } else {
                    for (int e = 0; e < len; ++e) {
                        const float u = coef * gt[e];
                        ah[e] = (awritten[ia] ? ah[e] : 0.0f) + u * qq[e];
                        qh[e] = (qwritten[iq] ? qh[e] : 0.0f) + u * a[e];
                    }
                    awritten[ia] = qwritten[iq] = 1;
                }
            }
            // the own term a_t q_0: its adjoint of a_t completes a-hat_t, which goes to G in place
            const float* at = A + (size_t)t * M + start;
            const float* ah = ahat + (size_t)t * C;
            float* qh0 = qhat;
            if (qwritten[0]) {
                if (awritten[t])
                    for (int e = 0; e < len; ++e) {
                        qh0[e] += gt[e] * at[e];
                        gt[e] = ah[e] + gt[e] * q[e];
                    }
                else
                    for (int e = 0; e < len; ++e) {
                        qh0[e] += gt[e] * at[e];
                        gt[e] = gt[e] * q[e];
                    }
            } else {
                if (awritten[t])
                    for (int e = 0; e < len; ++e) {
                        qh0[e] = gt[e] * at[e];
                        gt[e] = ah[e] + gt[e] * q[e];
                    }
                else
                    for (int e = 0; e < len; ++e) {
                        qh0[e] = gt[e] * at[e];
                        gt[e] = gt[e] * q[e];
                    }
                qwritten[0] = 1;
            }
        }
        for (int g = 0; g < S; ++g) {  // q_g = -sum s_b s_c: its adjoint reaches every s_b
            if (T.deg[g] != d - 1 || !qwritten[g]) continue;
            const float* qh = qhat + (size_t)g * C;
            for (int k = T.ustart[g]; k < T.ustart[g + 1]; ++k) {
                const int b = T.u_b[k], c = T.u_c[k];
                float* gb = G + (size_t)b * M + start;
                const float* hc = H + (size_t)c * M + start;
                for (int e = 0; e < len; ++e) gb[e] -= 2.0f * qh[e] * hc[e];
                if (b != c) {
                    float* gc = G + (size_t)c * M + start;
                    const float* hb = H + (size_t)b * M + start;
                    for (int e = 0; e < len; ++e) gc[e] -= 2.0f * qh[e] * hb[e];
                }
            }
        }
    }
    float* g0 = G + start;
    for (int e = 0; e < len; ++e) g0[e] *= q[e];  // s_0 = tanh(a_0)
}

// --------------------------------------------------------------------- layer normalization
// One row and its coefficient streams. c = a - mean(a) per stream, v = mean(sum c_b c_c) (+ eps
// at 0), r = v^-1/2 by its recurrence, z = c r by the product rule, y = gamma z + beta.
// The state of row n lives at c + n S w, z + n S w, v + n S, r + n S of the layer's region.
struct NormState {
    float* c;
    float* z;
    float* v;
    float* r;
    NormState(float* base, int N, int S, int w)
        : c(base), z(base + (size_t)N * S * w), v(z + (size_t)N * S * w), r(v + (size_t)N * S) {}
    NormState(const float* base, int N, int S, int w) : NormState(const_cast<float*>(base), N, S, w) {}
};

// Computes the state of one row from its value stream a (step: the distance between streams).
void norm_row(const Tables& T, const float* a, size_t step, int w, float eps, float* c, float* z,
              float* v, float* r) {
    const int S = T.S;
    const float inv = 1.0f / w;
    for (int s = 0; s < S; ++s) {
        const float* as = a + (size_t)s * step;
        float* cs = c + (size_t)s * w;
        float m = 0.0f;
        for (int k = 0; k < w; ++k) m += as[k];
        m *= inv;
        for (int k = 0; k < w; ++k) cs[k] = as[k] - m;
    }
    for (int s = 0; s < S; ++s) {
        float acc = 0.0f;
        for (int p = T.ustart[s]; p < T.ustart[s + 1]; ++p) {
            const float* cb = c + (size_t)T.u_b[p] * w;
            const float* cc = c + (size_t)T.u_c[p] * w;
            float dot = 0.0f;
            for (int k = 0; k < w; ++k) dot += cb[k] * cc[k];
            acc += T.u_f[p] * dot;
        }
        v[s] = acc * inv;
    }
    v[0] += eps;
    r[0] = 1.0f / std::sqrt(v[0]);
    for (int s = 1; s < S; ++s) {
        float num = 0.0f;
        for (int p = T.rstart[s]; p < T.rstart[s + 1]; ++p) num += T.r_coef[p] * r[T.r_r[p]] * v[T.r_v[p]];
        r[s] = num / (v[0] * (float)T.rdiv[s]);
    }
    for (int s = 0; s < S; ++s) {
        float* zs = z + (size_t)s * w;
        const float r0 = r[0];
        const float* cs = c + (size_t)s * w;
        for (int k = 0; k < w; ++k) zs[k] = cs[k] * r0;  // the pair (s, 0)
        for (int p = T.pstart[s]; p < T.pstart[s + 1]; ++p) {
            if (T.p_c[p] == 0) continue;
            const float* cb = c + (size_t)T.p_b[p] * w;
            const float rc = r[T.p_c[p]];
            for (int k = 0; k < w; ++k) zs[k] += cb[k] * rc;
        }
    }
}

// Y streams (N rows) = gamma z + beta from the A streams; A and Y are (S N x w). Saves the state.
void norm_forward(const Tables& T, const float* A, float* Y, int N, int w, const float* gamma,
                  const float* beta, float eps, float* state) {
    const int S = T.S;
    const size_t step = (size_t)N * w;
    NormState st(state, N, S, w);
    for (int n = 0; n < N; ++n) {
        const float* a = A + (size_t)n * w;
        float* y = Y + (size_t)n * w;
        float* z = st.z + (size_t)n * S * w;
        norm_row(T, a, step, w, eps, st.c + (size_t)n * S * w, z, st.v + (size_t)n * S, st.r + (size_t)n * S);
        for (int k = 0; k < w; ++k) y[k] = gamma[k] * z[k] + beta[k];
        for (int s = 1; s < S; ++s) {
            float* ys = y + (size_t)s * step;
            const float* zs = z + (size_t)s * w;
            for (int k = 0; k < w; ++k) ys[k] = gamma[k] * zs[k];
        }
    }
}

// G holds the adjoints of the y streams on entry and of the a streams on exit;
// ggamma and gbeta (w each) are overwritten with the gradients of gamma and beta.
void norm_backward(const Tables& T, const float* state, float* G, int N, int w, const float* gamma,
                   float* ggamma, float* gbeta) {
    const int S = T.S;
    const size_t step = (size_t)N * w;
    const float inv = 1.0f / w;
    NormState st(state, N, S, w);
    std::vector<float> zhat((size_t)S * w), chat((size_t)S * w), rhat(S), vhat(S);
    for (int k = 0; k < w; ++k) ggamma[k] = gbeta[k] = 0.0f;
    for (int n = 0; n < N; ++n) {
        float* g = G + (size_t)n * w;
        const float* c = st.c + (size_t)n * S * w;
        const float* z = st.z + (size_t)n * S * w;
        const float* v = st.v + (size_t)n * S;
        const float* r = st.r + (size_t)n * S;
        for (int k = 0; k < w; ++k) gbeta[k] += g[k];
        for (int s = 0; s < S; ++s) {
            const float* gs = g + (size_t)s * step;
            const float* zs = z + (size_t)s * w;
            float* zh = zhat.data() + (size_t)s * w;
            for (int k = 0; k < w; ++k) {
                ggamma[k] += gs[k] * zs[k];
                zh[k] = gamma[k] * gs[k];
            }
            rhat[s] = vhat[s] = 0.0f;
        }
        for (int s = 0; s < S; ++s) {  // z_s = sum c_b r_c, the pair (s, 0) first
            const float* zh = zhat.data() + (size_t)s * w;
            float* ch = chat.data() + (size_t)s * w;
            const float* cs = c + (size_t)s * w;
            const float r0 = r[0];
            float acc = 0.0f;
            for (int k = 0; k < w; ++k) {
                ch[k] = zh[k] * r0;
                acc += zh[k] * cs[k];
            }
            rhat[0] += acc;
        }
        for (int s = 0; s < S; ++s) {
            const float* zh = zhat.data() + (size_t)s * w;
            for (int p = T.pstart[s]; p < T.pstart[s + 1]; ++p) {
                if (T.p_c[p] == 0) continue;
                float* cb = chat.data() + (size_t)T.p_b[p] * w;
                const float* cbv = c + (size_t)T.p_b[p] * w;
                const float rc = r[T.p_c[p]];
                float acc = 0.0f;
                for (int k = 0; k < w; ++k) {
                    cb[k] += zh[k] * rc;
                    acc += zh[k] * cbv[k];
                }
                rhat[T.p_c[p]] += acc;
            }
        }
        for (int s = S - 1; s >= 1; --s) {  // r_s = num / (v_0 rdiv), highest degree first
            const float scale = 1.0f / (v[0] * (float)T.rdiv[s]);
            const float gs = rhat[s];
            vhat[0] -= gs * r[s] / v[0];
            for (int p = T.rstart[s]; p < T.rstart[s + 1]; ++p) {
                rhat[T.r_r[p]] += gs * scale * T.r_coef[p] * v[T.r_v[p]];
                vhat[T.r_v[p]] += gs * scale * T.r_coef[p] * r[T.r_r[p]];
            }
        }
        vhat[0] += rhat[0] * (-0.5f * r[0] * r[0] * r[0]);
        for (int s = 0; s < S; ++s) {  // v_s = inv sum c_b c_c over the unordered pairs
            const float gv = 2.0f * vhat[s] * inv;
            for (int p = T.ustart[s]; p < T.ustart[s + 1]; ++p) {
                const int b = T.u_b[p], cc = T.u_c[p];
                float* chb = chat.data() + (size_t)b * w;
                const float* ccv = c + (size_t)cc * w;
                for (int k = 0; k < w; ++k) chb[k] += gv * ccv[k];
                if (b != cc) {
                    float* chc = chat.data() + (size_t)cc * w;
                    const float* cbv = c + (size_t)b * w;
                    for (int k = 0; k < w; ++k) chc[k] += gv * cbv[k];
                }
            }
        }
        for (int s = 0; s < S; ++s) {  // c_s = a_s - mean(a_s)
            float* gs = g + (size_t)s * step;
            const float* ch = chat.data() + (size_t)s * w;
            float m = 0.0f;
            for (int k = 0; k < w; ++k) m += ch[k];
            m *= inv;
            for (int k = 0; k < w; ++k) gs[k] = ch[k] - m;
        }
    }
}

}  // namespace

extern "C" {

// Floats of work storage the forward pass fills and the backward pass reads.
long pinn_jets_work_size(int N, int L, const int* sizes, int S, int with_norm) {
    const long rows = (long)S * N;
    return (long)(with_norm ? norm_state_offset(L - 1, L, rows, sizes) : act_offset(L - 1, L, rows, sizes));
}

// X: (N x sizes[0]); out: (S N x sizes[L]) row blocks per stream.
// S streams with multi-indices alpha (S x D, stream 0 the zero index, downward closed, sorted by
// degree) over the input columns cols (D). gamma, beta: per hidden layer the LayerNorm weight
// and bias, or null without normalization.
void pinn_jets_forward(const float* X, int N, int L, const int* sizes, const float* const* W,
                       const float* const* b, int S, int D, const int* cols, const int* alpha,
                       const float* const* gamma, const float* const* beta, float eps, float* work,
                       float* out) {
    const Tables T(S, D, alpha);
    const long rows = (long)S * N;
    const int d = sizes[0];
    float* h0 = work;
    std::memcpy(h0, X, sizeof(float) * (size_t)N * d);
    std::memset(h0 + (size_t)N * d, 0, sizeof(float) * (size_t)(rows - N) * d);
    for (int s = 1; s < S; ++s) {  // the inputs are linear: unit first-degree streams only
        if (T.deg[s] != 1) continue;
        int k = 0;
        while (alpha[(size_t)s * D + k] == 0) ++k;
        float* block = h0 + (size_t)s * N * d;
        const int col = cols[k];
        for (int r = 0; r < N; ++r) block[(size_t)r * d + col] = 1.0f;
    }
    const int C = chunk_size(S);
    std::vector<float> q((size_t)S * C);
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
            norm_forward(T, A, Y, N, width, gamma[l], beta[l], eps, work + norm_state_offset(l, L, rows, sizes));
            A = Y;
        }
        tanh_inplace(A, count);
        float* Hn = work + hin_offset(l + 1, rows, sizes);
        for (size_t start = 0; start < M; start += C) {
            const int len = (int)((M - start) < (size_t)C ? (M - start) : (size_t)C);
            tanh_forward_chunk(T, A, Hn, M, start, len, q.data(), C);
        }
    }
}

// g_out: adjoints of `out`. gW[l] (out x in) and gb[l] are overwritten with the gradients.
// scratch: 2 * S * N * (largest layer width) floats.
// ggamma[l], gbeta[l]: gradients of the LayerNorm parameters (unused without normalization).
void pinn_jets_backward(int N, int L, const int* sizes, const float* const* W, int S, int D,
                        const int* alpha, const float* const* gamma, float eps, const float* work,
                        const float* g_out, float* const* gW, float* const* gb, float* const* ggamma,
                        float* const* gbeta, float* scratch) {
    const Tables T(S, D, alpha);
    const long rows = (long)S * N;
    int widest = 0;
    for (int l = 0; l <= L; ++l) widest = sizes[l] > widest ? sizes[l] : widest;
    float* Gcur = scratch;
    float* Gnext = scratch + (size_t)rows * widest;
    std::memcpy(Gcur, g_out, sizeof(float) * (size_t)rows * sizes[L]);
    const int C = chunk_size(S);
    std::vector<float> q((size_t)S * C), ahat((size_t)S * C), qhat((size_t)S * C);
    std::vector<char> awritten(S), qwritten(S);
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
        for (size_t start = 0; start < M; start += C) {
            const int len = (int)((M - start) < (size_t)C ? (M - start) : (size_t)C);
            tanh_backward_chunk(T, A, H, Gnext, M, start, len, q.data(), ahat.data(), qhat.data(), C,
                                awritten.data(), qwritten.data());
        }
        if (gamma != nullptr)  // from the adjoints of the normalised streams to the pre-norm ones
            norm_backward(T, work + norm_state_offset(l - 1, L, rows, sizes), Gnext, N, in, gamma[l - 1],
                          ggamma[l - 1], gbeta[l - 1]);
        float* tmp = Gcur;
        Gcur = Gnext;
        Gnext = tmp;
    }
}

}  // extern "C"
