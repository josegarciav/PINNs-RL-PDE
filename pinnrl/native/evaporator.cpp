// The falling-film evaporator's method-of-lines model integrated in C++ (experiment I16).
//
// The model is FallingFilmEvaporator.mol_rhs: first-order upwind in z for the liquid flow W and
// the solids fraction X,
//
//     dW_i/dt = -v (W_i - W_{i-1}) / dz - P q_i / lambda
//     dX_i/dt = -v (X_i - X_{i-1}) / dz + X_i P q_i / (lambda W_i)
//
// Here P is the source coefficient the caller passes, the film velocity times the wetted
// perimeter (m^2/s): the balance per unit length is (W/v)_t + W_z = -(perimeter) q / lambda.
//     q_i     = (T_s(t) - T_sat - k_bpe X_i) / (1/U + R_f(t))
//
// with the inlet (W_0, X_0) = (W_in(t), X_in(t)) a boundary value. The inputs W_in, X_in, T_s and
// R_f are tables on one time grid, interpolated by a natural cubic spline.
//
// Time integration is the L-stable SDIRK method of order 4 with 5 stages and gamma = 1/4
// (Hairer & Wanner, Solving ODEs II, table IV.6.5) with its embedded third-order estimate. The
// Jacobian of the upwind scheme is block lower-bidiagonal (cell i depends on itself and on
// i - 1), so each implicit stage is solved exactly by one sweep downstream with a 2 x 2 Newton
// iteration per cell: no global Newton, no factorisation, O(nz) per stage. The error estimate is
// filtered by (I - h gamma J)^-1, the same sweep made linear, so that stiff components do not
// make it pessimistic. Float64 throughout.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

const double GAMMA = 0.25;
const double C[5] = {0.25, 0.75, 11.0 / 20.0, 0.5, 1.0};
const double A[5][5] = {
    {0.25, 0, 0, 0, 0},
    {0.5, 0.25, 0, 0, 0},
    {17.0 / 50.0, -1.0 / 25.0, 0.25, 0, 0},
    {371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, 0.25, 0},
    {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25},
};
const double B[5] = {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25};
const double BHAT[5] = {59.0 / 48.0, -17.0 / 96.0, 225.0 / 32.0, -85.0 / 12.0, 0.0};

// The four input signals on one time grid, interpolated by a natural cubic spline (C^2, so the
// integrator keeps its order between knots; a two-point table is the linear ramp the filters use)
// and held constant outside the grid.
struct Signals {
    int n = 0;
    const double* t = nullptr;
    const double* v = nullptr;         // n x 4: W_in, X_in, T_s, R_f
    std::vector<double> m;             // n x 4 second derivatives of the spline

    void prepare() {
        m.assign((size_t)n * 4, 0.0);
        if (n < 3) return;
        std::vector<double> c(n), d(n), rhs(n);
        for (int k = 0; k < 4; ++k) {
            // natural spline: tridiagonal system for the second derivatives at the knots
            for (int i = 1; i < n - 1; ++i) {
                const double h0 = t[i] - t[i - 1], h1 = t[i + 1] - t[i];
                const double s0 = (v[(size_t)i * 4 + k] - v[(size_t)(i - 1) * 4 + k]) / h0;
                const double s1 = (v[(size_t)(i + 1) * 4 + k] - v[(size_t)i * 4 + k]) / h1;
                c[i] = 2.0 * (h0 + h1);
                d[i] = h1;
                rhs[i] = 6.0 * (s1 - s0);
            }
            for (int i = 2; i < n - 1; ++i) {  // forward elimination (sub-diagonal is h0 = d[i-1])
                const double w = (t[i] - t[i - 1]) / c[i - 1];
                c[i] -= w * d[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }
            double* mk = m.data();
            mk[(size_t)(n - 2) * 4 + k] = rhs[n - 2] / c[n - 2];
            for (int i = n - 3; i >= 1; --i)
                mk[(size_t)i * 4 + k] = (rhs[i] - d[i] * mk[(size_t)(i + 1) * 4 + k]) / c[i];
        }
    }

    void at(double tau, double out[4]) const {
        if (n == 1 || tau <= t[0]) {
            for (int k = 0; k < 4; ++k) out[k] = v[k];
            return;
        }
        if (tau >= t[n - 1]) {
            for (int k = 0; k < 4; ++k) out[k] = v[(size_t)(n - 1) * 4 + k];
            return;
        }
        int lo = 0, hi = n - 1;  // t[lo] <= tau < t[hi]
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (t[mid] <= tau) lo = mid; else hi = mid;
        }
        const double h = t[hi] - t[lo];
        const double a = (t[hi] - tau) / h, b = (tau - t[lo]) / h;
        const double h2 = h * h / 6.0;
        for (int k = 0; k < 4; ++k) {
            const double ylo = v[(size_t)lo * 4 + k], yhi = v[(size_t)hi * 4 + k];
            const double mlo = m[(size_t)lo * 4 + k], mhi = m[(size_t)hi * 4 + k];
            out[k] = a * ylo + b * yhi + ((a * a * a - a) * mlo + (b * b * b - b) * mhi) * h2;
        }
    }
};

struct Model {
    int nz;           // grid points including the inlet; nz - 1 cells carry state
    double dz, v, P, U, lam, T_sat, k;
    Signals sig;
    int m() const { return nz - 1; }

    // f(t, y) for the interleaved state y = (W_1, X_1, W_2, X_2, ...).
    void rhs(double tau, const double* y, double* dy) const {
        double s[4];
        sig.at(tau, s);
        const double D = 1.0 / U + s[3];
        double Wu = s[0], Xu = s[1];
        for (int i = 0; i < m(); ++i) {
            const double W = y[2 * i], X = y[2 * i + 1];
            const double q = (s[2] - T_sat - k * X) / D;
            dy[2 * i] = -v * (W - Wu) / dz - P * q / lam;
            dy[2 * i + 1] = -v * (X - Xu) / dz + X * P * q / (lam * W);
            Wu = W;
            Xu = X;
        }
    }

    // Solves Y = r + h gamma f(tau, Y) cell by cell; returns false when Newton fails.
    bool stage(double tau, double hg, const double* r, const double* guess, double* Y, double rtol,
               double atol, long* newton) const {
        double s[4];
        sig.at(tau, s);
        const double D = 1.0 / U + s[3];
        const double a = v / dz;
        double Wu = s[0], Xu = s[1];
        for (int i = 0; i < m(); ++i) {
            double W = guess[2 * i], X = guess[2 * i + 1];
            const double rW = r[2 * i], rX = r[2 * i + 1];
            bool ok = false;
            for (int it = 0; it < 25; ++it) {
                ++*newton;
                if (!(W > 0.0)) return false;
                const double q = (s[2] - T_sat - k * X) / D;
                const double src = P * q / (lam * W);  // X's source per unit X
                const double gW = W - hg * (-a * (W - Wu) - P * q / lam) - rW;
                const double gX = X - hg * (-a * (X - Xu) + X * src) - rX;
                const double j11 = 1.0 + hg * a;
                const double j12 = -hg * (P * k / (lam * D));
                const double j21 = hg * (X * src / W);
                const double j22 = 1.0 - hg * (-a + src - X * P * k / (lam * W * D));
                const double det = j11 * j22 - j12 * j21;
                const double dW = (gW * j22 - j12 * gX) / det;
                const double dX = (j11 * gX - j21 * gW) / det;
                W -= dW;
                X -= dX;
                if (std::fabs(dW) <= 1e-3 * (atol + rtol * std::fabs(W)) &&
                    std::fabs(dX) <= 1e-3 * (atol + rtol * std::fabs(X))) {
                    ok = true;
                    break;
                }
            }
            if (!ok || !(W > 0.0)) return false;
            Y[2 * i] = W;
            Y[2 * i + 1] = X;
            Wu = W;
            Xu = X;
        }
        return true;
    }

    // e <- (I - hg J(tau, y))^-1 e, J the Jacobian of f, by the same downstream sweep.
    void filter(double tau, double hg, const double* y, double* e) const {
        double s[4];
        sig.at(tau, s);
        const double D = 1.0 / U + s[3];
        const double a = v / dz;
        double eWu = 0.0, eXu = 0.0;  // the inlet does not move
        for (int i = 0; i < m(); ++i) {
            const double W = y[2 * i], X = y[2 * i + 1];
            const double q = (s[2] - T_sat - k * X) / D;
            const double src = P * q / (lam * W);
            const double j11 = 1.0 + hg * a;
            const double j12 = -hg * (P * k / (lam * D));
            const double j21 = hg * (X * src / W);
            const double j22 = 1.0 - hg * (-a + src - X * P * k / (lam * W * D));
            const double bW = e[2 * i] + hg * a * eWu;
            const double bX = e[2 * i + 1] + hg * a * eXu;
            const double det = j11 * j22 - j12 * j21;
            eWu = (bW * j22 - j12 * bX) / det;
            eXu = (j11 * bX - j21 * bW) / det;
            e[2 * i] = eWu;
            e[2 * i + 1] = eXu;
        }
    }
};

}  // namespace

extern "C" {

// Integrates the model from s0 at t_out[0] and writes the state at every t_out into out
// (nt x 2 nz, W_0..W_{nz-1} then X_0..X_{nz-1}, the inlet entries from the signals).
// geom: L, v, P, U, lambda, T_sat, k_bpe. Signals: n_sig rows of (t) and (W_in, X_in, T_s, R_f).
// stats: accepted steps, rejected steps, Newton iterations.
// Returns 0 on success, 1 when the step size underflows, 2 when Newton fails at the smallest step.
int evap_solve(int nz, const double* geom, int n_sig, const double* sig_t, const double* sig_v,
               const double* s0, int nt, const double* t_out, double rtol, double atol, double h0,
               double* out, long* stats) {
    Model md;
    md.nz = nz;
    md.dz = geom[0] / (nz - 1);
    md.v = geom[1];
    md.P = geom[2];
    md.U = geom[3];
    md.lam = geom[4];
    md.T_sat = geom[5];
    md.k = geom[6];
    md.sig.n = n_sig;
    md.sig.t = sig_t;
    md.sig.v = sig_v;
    md.sig.prepare();
    const int m = md.m(), n = 2 * m;
    std::vector<double> y(n), ynew(n), r(n), Y(n), e(n), guess(n);
    std::vector<std::vector<double>> k(5, std::vector<double>(n));
    for (int i = 0; i < m; ++i) {
        y[2 * i] = s0[1 + i];
        y[2 * i + 1] = s0[nz + 1 + i];
    }
    long accepted = 0, rejected = 0, newton = 0;
    bool just_rejected = false;
    auto store = [&](int row, double tau, const double* state) {
        double s[4];
        md.sig.at(tau, s);
        double* o = out + (size_t)row * 2 * nz;
        o[0] = s[0];
        o[nz] = s[1];
        for (int i = 0; i < m; ++i) {
            o[1 + i] = state[2 * i];
            o[nz + 1 + i] = state[2 * i + 1];
        }
    };
    double t = t_out[0];
    store(0, t, y.data());
    const double span = t_out[nt - 1] - t_out[0];
    double h = h0 > 0.0 ? h0 : 1e-3 * span;
    if (span <= 0.0) {
        for (int row = 1; row < nt; ++row) store(row, t_out[row], y.data());
        return 0;
    }
    const double h_min = 1e-12 * span;
    for (int row = 1; row < nt; ++row) {
        const double t_end = t_out[row];
        while (t < t_end) {
            bool last = false;
            if (t + h >= t_end - 1e-12 * span) {
                h = t_end - t;
                last = true;
            }
            const double hg = h * GAMMA;
            bool ok = true;
            for (int st = 0; st < 5 && ok; ++st) {
                for (int j = 0; j < n; ++j) {
                    double acc = y[j];
                    for (int q = 0; q < st; ++q) acc += h * A[st][q] * k[q][j];
                    r[j] = acc;
                    // predictor: the previous stage's slope carried on (the first stage from y)
                    guess[j] = st == 0 ? y[j] : acc + hg * k[st - 1][j];
                }
                ok = md.stage(t + C[st] * h, hg, r.data(), guess.data(), Y.data(), rtol, atol, &newton);
                if (!ok) break;
                for (int j = 0; j < n; ++j) k[st][j] = (Y[j] - r[j]) / hg;
            }
            double err = 0.0;
            if (ok) {
                for (int j = 0; j < n; ++j) {
                    ynew[j] = Y[j];  // stiffly accurate: the last stage is the step
                    double acc = 0.0;
                    for (int st = 0; st < 5; ++st) acc += (B[st] - BHAT[st]) * k[st][j];
                    e[j] = h * acc;
                }
                md.filter(t + h, hg, ynew.data(), e.data());
                for (int j = 0; j < n; ++j) {
                    const double sc = atol + rtol * std::max(std::fabs(y[j]), std::fabs(ynew[j]));
                    err += (e[j] / sc) * (e[j] / sc);
                }
                err = std::sqrt(err / n);
            }
            if (ok && err <= 1.0) {
                t = last ? t_end : t + h;
                y.swap(ynew);
                ++accepted;
                double factor = err < 1e-10 ? 4.0 : std::min(4.0, std::max(0.2, 0.9 * std::pow(err, -0.25)));
                if (just_rejected) factor = std::min(factor, 1.0);  // no growth right after a rejection
                just_rejected = false;
                h = std::min(h * factor, span);
            } else {
                ++rejected;
                just_rejected = true;
                h *= ok ? std::max(0.2, 0.9 * std::pow(err, -0.25)) : 0.25;
                if (h < h_min) {
                    stats[0] = accepted;
                    stats[1] = rejected;
                    stats[2] = newton;
                    return ok ? 1 : 2;
                }
            }
        }
        store(row, t_end, y.data());
    }
    stats[0] = accepted;
    stats[1] = rejected;
    stats[2] = newton;
    return 0;
}

}  // extern "C"
