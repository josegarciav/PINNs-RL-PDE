// I11: a PINN training step with no framework, in C++.
//
// The heat problem of bench_python.py: u_t = alpha u_xx, a tanh MLP 2-64-64-64-64-1 in float32,
// loss = mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2), Adam. The derivatives u_x, u_xx, u_t are
// carried forward through the layers (Taylor mode) and the weight gradient is one hand-written
// reverse pass over those four streams. Two kernels: `naive` (plain loops the compiler
// vectorises) and `blas` (Accelerate's sgemm and vvtanhf, the library PyTorch calls on macOS).
//
// Build: clang++ -O3 -mcpu=native -std=c++17 pinn.cpp -framework Accelerate -o pinn_cpp
// Run:   ./pinn_cpp --mode blas --steps 3000 --seed 0
//        ./pinn_cpp --mode blas --check in.bin out.bin     (gradient check against PyTorch)

#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::vector;
typedef vector<float> Vec;

static const float ALPHA = 0.1f;
static const int WIDTH = 64, DEPTH = 4, LAYERS = DEPTH + 1;
static const int N_COL = 4096, N_IC = 512, N_BC = 512, N_VAL = N_IC + N_BC;
static const float W_IC = 10.0f, W_BC = 10.0f, LR = 1e-3f;
static const float PI = 3.14159265358979323846f;
static bool USE_BLAS = true;
static double T_GEMM = 0.0, T_TANH = 0.0;  // time inside the matrix products and tanh

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------ kernels
// C (n x m) = A (n x k) * B^T, B (m x k), with Bt (k x m) its transpose for the naive kernel.
static void mul_bt_impl(const float* A, const float* B, const float* Bt, float* C, int n, int k, int m) {
    if (USE_BLAS) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, k, 1.0f, A, k, B, k, 0.0f, C, m);
        return;
    }
    std::memset(C, 0, sizeof(float) * (size_t)n * m);
    for (int i = 0; i < n; ++i) {
        float* c = C + (size_t)i * m;
        const float* a = A + (size_t)i * k;
        for (int p = 0; p < k; ++p) {
            const float v = a[p];
            const float* b = Bt + (size_t)p * m;
            for (int j = 0; j < m; ++j) c[j] += v * b[j];
        }
    }
}

// C (n x k) = G (n x m) * W (m x k)
static void mul_w_impl(const float* G, const float* W, float* C, int n, int m, int k) {
    if (USE_BLAS) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, m, 1.0f, G, m, W, k, 0.0f, C, k);
        return;
    }
    std::memset(C, 0, sizeof(float) * (size_t)n * k);
    for (int i = 0; i < n; ++i) {
        float* c = C + (size_t)i * k;
        const float* g = G + (size_t)i * m;
        for (int j = 0; j < m; ++j) {
            const float v = g[j];
            const float* w = W + (size_t)j * k;
            for (int p = 0; p < k; ++p) c[p] += v * w[p];
        }
    }
}

// gW (m x k) += G^T (n x m) * H (n x k)
static void add_gt_h_impl(const float* G, const float* H, float* gW, int n, int m, int k) {
    if (USE_BLAS) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, k, n, 1.0f, G, m, H, k, 1.0f, gW, k);
        return;
    }
    for (int i = 0; i < n; ++i) {
        const float* g = G + (size_t)i * m;
        const float* h = H + (size_t)i * k;
        for (int j = 0; j < m; ++j) {
            const float v = g[j];
            float* w = gW + (size_t)j * k;
            for (int p = 0; p < k; ++p) w[p] += v * h[p];
        }
    }
}

static void tanh_impl(float* x, int n) {
    if (USE_BLAS) {
        vvtanhf(x, x, &n);
        return;
    }
    for (int i = 0; i < n; ++i) x[i] = std::tanh(x[i]);
}

static void mul_bt(const float* A, const float* B, const float* Bt, float* C, int n, int k, int m) {
    const double t = now(); mul_bt_impl(A, B, Bt, C, n, k, m); T_GEMM += now() - t;
}
static void mul_w(const float* G, const float* W, float* C, int n, int m, int k) {
    const double t = now(); mul_w_impl(G, W, C, n, m, k); T_GEMM += now() - t;
}
static void add_gt_h(const float* G, const float* H, float* gW, int n, int m, int k) {
    const double t = now(); add_gt_h_impl(G, H, gW, n, m, k); T_GEMM += now() - t;
}
static void tanh_inplace(float* x, int n) {
    const double t = now(); tanh_impl(x, n); T_TANH += now() - t;
}

// ------------------------------------------------------------------ network
struct Net {
    int in[LAYERS], out[LAYERS];
    Vec W[LAYERS], b[LAYERS], Wt[LAYERS];
    Vec gW[LAYERS], gb[LAYERS];
    Vec mW[LAYERS], vW[LAYERS], mb[LAYERS], vb[LAYERS];
    Net() {
        for (int l = 0; l < LAYERS; ++l) {
            in[l] = l == 0 ? 2 : WIDTH;
            out[l] = l == LAYERS - 1 ? 1 : WIDTH;
            size_t n = (size_t)in[l] * out[l];
            W[l].assign(n, 0); Wt[l].assign(n, 0); gW[l].assign(n, 0); mW[l].assign(n, 0); vW[l].assign(n, 0);
            b[l].assign(out[l], 0); gb[l].assign(out[l], 0); mb[l].assign(out[l], 0); vb[l].assign(out[l], 0);
        }
    }
    void transpose() {
        for (int l = 0; l < LAYERS; ++l)
            for (int j = 0; j < out[l]; ++j)
                for (int p = 0; p < in[l]; ++p) Wt[l][(size_t)p * out[l] + j] = W[l][(size_t)j * in[l] + p];
    }
    void zero_grad() {
        for (int l = 0; l < LAYERS; ++l) {
            std::fill(gW[l].begin(), gW[l].end(), 0.0f);
            std::fill(gb[l].begin(), gb[l].end(), 0.0f);
        }
    }
};

struct Rng {  // xoshiro256++
    uint64_t s[4];
    explicit Rng(uint64_t seed) {
        for (auto& v : s) {  // splitmix64
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            v = z ^ (z >> 31);
        }
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        uint64_t r = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    float uniform() { return (float)(next() >> 40) * (1.0f / 16777216.0f); }
};

// Work arrays. The four streams (value, x-slope, x-curvature, t-slope) of the collocation batch
// are stacked as the rows [0,N), [N,2N), [2N,3N), [3N,4N) of one matrix, so each layer is one product.
struct Work {
    Vec Hin[LAYERS];   // (4N x in[l]) inputs of layer l
    Vec A[LAYERS];     // (4N x out[l]) pre-activations: a, a_x, a_xx, a_t; the a block holds tanh(a) after the forward pass
    Vec G, G2;         // adjoints
    Vec Vin[LAYERS], VA[LAYERS], VG, VG2;  // the value-only batch (initial and boundary points)
    Work() {
        for (int l = 0; l < LAYERS; ++l) {
            int in = l == 0 ? 2 : WIDTH, out = l == LAYERS - 1 ? 1 : WIDTH;
            Hin[l].assign((size_t)4 * N_COL * in, 0);
            A[l].assign((size_t)4 * N_COL * out, 0);
            Vin[l].assign((size_t)N_VAL * in, 0);
            VA[l].assign((size_t)N_VAL * out, 0);
        }
        G.assign((size_t)4 * N_COL * WIDTH, 0); G2 = G;
        VG.assign((size_t)N_VAL * WIDTH, 0); VG2 = VG;
    }
};

// Residual loss mean(r^2) on the collocation batch xt (N x 2); accumulates the weight gradient.
static float residual_pass(Net& net, Work& w, const float* xt) {
    const int N = N_COL;
    float* h0 = w.Hin[0].data();
    for (int i = 0; i < N; ++i) {
        h0[2 * i] = xt[2 * i]; h0[2 * i + 1] = xt[2 * i + 1];                 // value
        h0[2 * (N + i)] = 1.0f; h0[2 * (N + i) + 1] = 0.0f;                   // d/dx
        h0[2 * (2 * N + i)] = 0.0f; h0[2 * (2 * N + i) + 1] = 0.0f;           // d2/dx2
        h0[2 * (3 * N + i)] = 0.0f; h0[2 * (3 * N + i) + 1] = 1.0f;           // d/dt
    }
    for (int l = 0; l < LAYERS; ++l) {
        const int in = net.in[l], out = net.out[l];
        float* A = w.A[l].data();
        mul_bt(w.Hin[l].data(), net.W[l].data(), net.Wt[l].data(), A, 4 * N, in, out);
        const float* b = net.b[l].data();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < out; ++j) A[(size_t)i * out + j] += b[j];
        if (l == LAYERS - 1) break;
        tanh_inplace(A, N * out);
        float* H = w.Hin[l + 1].data();
        const size_t M = (size_t)N * out;
        const float *s = A, *ax = A + M, *axx = A + 2 * M, *at = A + 3 * M;
        float *h = H, *hx = H + M, *hxx = H + 2 * M, *ht = H + 3 * M;
        for (size_t i = 0; i < M; ++i) {
            const float d1 = 1.0f - s[i] * s[i], d2 = -2.0f * s[i] * d1;
            h[i] = s[i];
            hx[i] = d1 * ax[i];
            hxx[i] = d2 * ax[i] * ax[i] + d1 * axx[i];
            ht[i] = d1 * at[i];
        }
    }
    // output: r = u_t - alpha u_xx
    const float* out = w.A[LAYERS - 1].data();
    float* G = w.G.data();
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
        const float r = out[3 * N + i] - ALPHA * out[2 * N + i];
        loss += (double)r * r;
        const float g = 2.0f * r / N;
        G[i] = 0.0f; G[N + i] = 0.0f; G[2 * N + i] = -ALPHA * g; G[3 * N + i] = g;
    }
    float* Gcur = w.G.data();
    float* Gnext = w.G2.data();
    for (int l = LAYERS - 1; l >= 0; --l) {
        const int in = net.in[l], out_w = net.out[l];
        add_gt_h(Gcur, w.Hin[l].data(), net.gW[l].data(), 4 * N, out_w, in);
        float* gb = net.gb[l].data();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < out_w; ++j) gb[j] += Gcur[(size_t)i * out_w + j];
        if (l == 0) break;
        mul_w(Gcur, net.W[l].data(), Gnext, 4 * N, out_w, in);  // adjoints of h, h_x, h_xx, h_t
        const size_t M = (size_t)N * in;
        const float* A = w.A[l - 1].data();
        const float *s = A, *ax = A + M, *axx = A + 2 * M, *at = A + 3 * M;
        float *gh = Gnext, *ghx = Gnext + M, *ghxx = Gnext + 2 * M, *ght = Gnext + 3 * M;
        for (size_t i = 0; i < M; ++i) {
            const float d1 = 1.0f - s[i] * s[i], d2 = -2.0f * s[i] * d1;
            const float d3 = -2.0f * d1 * (1.0f - 3.0f * s[i] * s[i]);
            const float a = gh[i] * d1 + ghx[i] * d2 * ax[i] + ghxx[i] * (d3 * ax[i] * ax[i] + d2 * axx[i]) +
                            ght[i] * d2 * at[i];
            const float bx = ghx[i] * d1 + ghxx[i] * 2.0f * d2 * ax[i];
            gh[i] = a; ghx[i] = bx; ghxx[i] = ghxx[i] * d1; ght[i] = ght[i] * d1;
        }
        std::swap(Gcur, Gnext);
    }
    return (float)(loss / N);
}

// sum_i coef_i/2 * (u(x_i, t_i) - target_i)^2 on the value batch (rows: x, t, target, coef).
static float value_pass(Net& net, Work& w, const float* batch) {
    const int N = N_VAL;
    float* h0 = w.Vin[0].data();
    for (int i = 0; i < N; ++i) { h0[2 * i] = batch[4 * i]; h0[2 * i + 1] = batch[4 * i + 1]; }
    for (int l = 0; l < LAYERS; ++l) {
        const int in = net.in[l], out = net.out[l];
        float* A = w.VA[l].data();
        mul_bt(w.Vin[l].data(), net.W[l].data(), net.Wt[l].data(), A, N, in, out);
        const float* b = net.b[l].data();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < out; ++j) A[(size_t)i * out + j] += b[j];
        if (l == LAYERS - 1) break;
        tanh_inplace(A, N * out);
        std::memcpy(w.Vin[l + 1].data(), A, sizeof(float) * (size_t)N * out);
    }
    const float* u = w.VA[LAYERS - 1].data();
    float* Gcur = w.VG.data();
    float* Gnext = w.VG2.data();
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
        const float e = u[i] - batch[4 * i + 2], c = batch[4 * i + 3];
        loss += 0.5 * (double)c * e * e;
        Gcur[i] = c * e;
    }
    for (int l = LAYERS - 1; l >= 0; --l) {
        const int in = net.in[l], out_w = net.out[l];
        add_gt_h(Gcur, w.Vin[l].data(), net.gW[l].data(), N, out_w, in);
        float* gb = net.gb[l].data();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < out_w; ++j) gb[j] += Gcur[(size_t)i * out_w + j];
        if (l == 0) break;
        mul_w(Gcur, net.W[l].data(), Gnext, N, out_w, in);
        const float* s = w.VA[l - 1].data();
        const size_t M = (size_t)N * in;
        for (size_t i = 0; i < M; ++i) Gnext[i] *= 1.0f - s[i] * s[i];
        std::swap(Gcur, Gnext);
    }
    return (float)loss;
}

static void adam(Net& net, int step) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float c1 = 1.0f - std::pow(b1, (float)step), c2 = 1.0f - std::pow(b2, (float)step);
    auto update = [&](Vec& p, Vec& g, Vec& m, Vec& v) {
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = b1 * m[i] + (1.0f - b1) * g[i];
            v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            p[i] -= LR * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
        }
    };
    for (int l = 0; l < LAYERS; ++l) {
        update(net.W[l], net.gW[l], net.mW[l], net.vW[l]);
        update(net.b[l], net.gb[l], net.mb[l], net.vb[l]);
    }
}

static void sample(Rng& rng, Vec& xt, Vec& val) {
    for (int i = 0; i < 2 * N_COL; ++i) xt[i] = rng.uniform();
    for (int i = 0; i < N_IC; ++i) {  // u(x, 0) = sin(pi x)
        const float x = rng.uniform();
        val[4 * i] = x; val[4 * i + 1] = 0.0f; val[4 * i + 2] = std::sin(PI * x); val[4 * i + 3] = 2.0f * W_IC / N_IC;
    }
    for (int i = 0; i < N_BC; ++i) {  // u(0, t) = u(1, t) = 0
        float* v = &val[4 * (N_IC + i)];
        v[0] = (float)(i % 2); v[1] = rng.uniform(); v[2] = 0.0f; v[3] = 2.0f * W_BC / N_BC;
    }
}

static double rel_l2(Net& net, Work& w) {
    // the value pass on a 101 x 101 grid, in chunks of the value batch
    double num = 0.0, den = 0.0;
    Vec batch((size_t)4 * N_VAL, 0.0f);
    const int total = 101 * 101;
    for (int start = 0; start < total; start += N_VAL) {
        for (int i = 0; i < N_VAL; ++i) {
            const int idx = std::min(start + i, total - 1);
            batch[4 * i] = (float)(idx / 101) / 100.0f;
            batch[4 * i + 1] = (float)(idx % 101) / 100.0f;
        }
        Net scratch = net;  // value_pass accumulates gradients; keep the real ones clean
        value_pass(scratch, w, batch.data());
        const float* u = w.VA[LAYERS - 1].data();
        for (int i = 0; i < N_VAL && start + i < total; ++i) {
            const double x = batch[4 * i], t = batch[4 * i + 1];
            const double exact = std::exp(-ALPHA * PI * PI * t) * std::sin(PI * x);
            num += (u[i] - exact) * (u[i] - exact);
            den += exact * exact;
        }
    }
    return std::sqrt(num / den);
}

int main(int argc, char** argv) {
    int steps = 3000;
    uint64_t seed = 0;
    std::string mode = "blas", check_in, check_out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--steps") steps = std::atoi(argv[++i]);
        else if (a == "--seed") seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--mode") mode = argv[++i];
        else if (a == "--check") { check_in = argv[++i]; check_out = argv[++i]; }
    }
    USE_BLAS = mode == "blas";
    Net net;
    Work work;
    Vec xt((size_t)2 * N_COL), val((size_t)4 * N_VAL);

    if (!check_in.empty()) {  // weights and one batch from PyTorch in, loss and gradient out
        FILE* f = std::fopen(check_in.c_str(), "rb");
        if (!f) { std::perror("check input"); return 1; }
        size_t ok = 1;
        for (int l = 0; l < LAYERS; ++l) {
            ok &= std::fread(net.W[l].data(), sizeof(float), net.W[l].size(), f) == net.W[l].size();
            ok &= std::fread(net.b[l].data(), sizeof(float), net.b[l].size(), f) == net.b[l].size();
        }
        ok &= std::fread(xt.data(), sizeof(float), xt.size(), f) == xt.size();
        ok &= std::fread(val.data(), sizeof(float), val.size(), f) == val.size();
        std::fclose(f);
        if (!ok) { std::fprintf(stderr, "short check input\n"); return 1; }
        net.transpose();
        net.zero_grad();
        float loss = residual_pass(net, work, xt.data()) + value_pass(net, work, val.data());
        FILE* o = std::fopen(check_out.c_str(), "wb");
        std::fwrite(&loss, sizeof(float), 1, o);
        for (int l = 0; l < LAYERS; ++l) {
            std::fwrite(net.gW[l].data(), sizeof(float), net.gW[l].size(), o);
            std::fwrite(net.gb[l].data(), sizeof(float), net.gb[l].size(), o);
        }
        std::fclose(o);
        return 0;
    }

    Rng init(seed), rng(1000 + seed);
    for (int l = 0; l < LAYERS; ++l) {  // Xavier uniform, zero biases
        const float bound = std::sqrt(6.0f / (net.in[l] + net.out[l]));
        for (auto& v : net.W[l]) v = (2.0f * init.uniform() - 1.0f) * bound;
    }
    double t_sample = 0, t_resid = 0, t_value = 0, t_adam = 0;
    float loss = 0;
    const double tic = now();
    for (int step = 1; step <= steps; ++step) {
        double t0 = now();
        sample(rng, xt, val);
        double t1 = now();
        net.transpose();
        net.zero_grad();
        loss = residual_pass(net, work, xt.data());
        double t2 = now();
        loss += value_pass(net, work, val.data());
        double t3 = now();
        adam(net, step);
        double t4 = now();
        t_sample += t1 - t0; t_resid += t2 - t1; t_value += t3 - t2; t_adam += t4 - t3;
    }
    const double elapsed = now() - tic;
    net.transpose();  // the evaluation below uses the final weights in either kernel
    const double k = 1e3 / steps;
    const double gemm_ms = T_GEMM * k, tanh_ms = T_TANH * k;
    std::printf(
        "{\"impl\": \"cpp-%s\", \"seed\": %llu, \"steps\": %d, \"ms_per_step\": %.4f, \"rel_l2\": %.6e, \"loss\": %.6e, "
        "\"stages_ms\": {\"sample\": %.4f, \"residual forward and backward\": %.4f, \"initial and boundary\": %.4f, "
        "\"optimizer\": %.4f}, \"kernels_ms\": {\"matrix products\": %.4f, \"tanh\": %.4f}}\n",
        mode.c_str(), (unsigned long long)seed, steps, elapsed * k, rel_l2(net, work), (double)loss, t_sample * k,
        t_resid * k, t_value * k, t_adam * k, gemm_ms, tanh_ms);
    return 0;
}
