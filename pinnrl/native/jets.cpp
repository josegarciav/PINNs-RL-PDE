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
// Streams: 0 is the value; 1..nf the first derivatives (input column first[i]); nf+1..nf+ns the
// second derivatives, parent[j] giving the index (0..nf-1) of the first derivative they extend.
// Float32, row-major; W[l] is (out x in) as in torch.nn.Linear.

#include <Accelerate/Accelerate.h>
#include <cstddef>
#include <cstring>

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

}  // namespace

extern "C" {

// Floats of work storage the forward pass fills and the backward pass reads.
long pinn_jets_work_size(int N, int L, const int* sizes, int S) {
    const long rows = (long)S * N;
    return (long)(act_offset(L - 1, L, rows, sizes));
}

// out: (S N x sizes[L]) row blocks per stream.
void pinn_jets_forward(const float* X, int N, int L, const int* sizes, const float* const* W,
                       const float* const* b, int nf, const int* first, int ns, const int* parent,
                       float* work, float* out) {
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
void pinn_jets_backward(int N, int L, const int* sizes, const float* const* W, int nf, int ns,
                        const int* parent, const float* work, const float* g_out, float* const* gW,
                        float* const* gb, float* scratch) {
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
        const float* A = work + act_offset(l - 1, L, rows, sizes);
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
        float* tmp = Gcur;
        Gcur = Gnext;
        Gnext = tmp;
    }
}

}  // extern "C"
