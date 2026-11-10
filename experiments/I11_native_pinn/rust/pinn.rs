// I11: a PINN training step with no framework, in Rust (the standard library only).
//
// The same program as cpp/pinn.cpp, line for line where the languages allow: the heat problem
// u_t = alpha u_xx, a tanh MLP 2-64-64-64-64-1 in f32, Taylor-mode forward derivatives, one
// hand-written reverse pass, Adam, the same random generator. Two kernels: `naive` (safe slice
// loops the compiler vectorises) and `blas` (Accelerate's sgemm and vvtanhf through FFI).
//
// Build: rustc -C opt-level=3 -C target-cpu=native -C panic=abort pinn.rs -o pinn_rust
// Run:   ./pinn_rust --mode blas --steps 3000 --seed 0
//        ./pinn_rust --mode blas --check in.bin out.bin

use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

const ALPHA: f32 = 0.1;
const WIDTH: usize = 64;
const DEPTH: usize = 4;
const LAYERS: usize = DEPTH + 1;
const N_COL: usize = 4096;
const N_IC: usize = 512;
const N_BC: usize = 512;
const N_VAL: usize = N_IC + N_BC;
const W_IC: f32 = 10.0;
const W_BC: f32 = 10.0;
const LR: f32 = 1e-3;
const PI: f32 = std::f32::consts::PI;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32,
        lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32,
    );
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
}
const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

// ------------------------------------------------------------------ kernels
/// C (n x m) = A (n x k) * B^T, B (m x k), with `bt` (k x m) its transpose for the naive kernel.
fn mul_bt(blas: bool, a: &[f32], b: &[f32], bt: &[f32], c: &mut [f32], n: usize, k: usize, m: usize) {
    if blas {
        unsafe {
            cblas_sgemm(ROW_MAJOR, NO_TRANS, TRANS, n as i32, m as i32, k as i32, 1.0, a.as_ptr(),
                k as i32, b.as_ptr(), k as i32, 0.0, c.as_mut_ptr(), m as i32);
        }
        return;
    }
    let c = &mut c[..n * m];
    c.fill(0.0);
    for (crow, arow) in c.chunks_exact_mut(m).zip(a[..n * k].chunks_exact(k)) {
        for (&v, brow) in arow.iter().zip(bt.chunks_exact(m)) {
            for (cj, &bj) in crow.iter_mut().zip(brow) {
                *cj += v * bj;
            }
        }
    }
}

/// C (n x k) = G (n x m) * W (m x k)
fn mul_w(blas: bool, g: &[f32], w: &[f32], c: &mut [f32], n: usize, m: usize, k: usize) {
    if blas {
        unsafe {
            cblas_sgemm(ROW_MAJOR, NO_TRANS, NO_TRANS, n as i32, k as i32, m as i32, 1.0, g.as_ptr(),
                m as i32, w.as_ptr(), k as i32, 0.0, c.as_mut_ptr(), k as i32);
        }
        return;
    }
    let c = &mut c[..n * k];
    c.fill(0.0);
    for (crow, grow) in c.chunks_exact_mut(k).zip(g[..n * m].chunks_exact(m)) {
        for (&v, wrow) in grow.iter().zip(w.chunks_exact(k)) {
            for (cp, &wp) in crow.iter_mut().zip(wrow) {
                *cp += v * wp;
            }
        }
    }
}

/// gW (m x k) += G^T (n x m) * H (n x k)
fn add_gt_h(blas: bool, g: &[f32], h: &[f32], gw: &mut [f32], n: usize, m: usize, k: usize) {
    if blas {
        unsafe {
            cblas_sgemm(ROW_MAJOR, TRANS, NO_TRANS, m as i32, k as i32, n as i32, 1.0, g.as_ptr(),
                m as i32, h.as_ptr(), k as i32, 1.0, gw.as_mut_ptr(), k as i32);
        }
        return;
    }
    for (grow, hrow) in g[..n * m].chunks_exact(m).zip(h[..n * k].chunks_exact(k)) {
        for (&v, wrow) in grow.iter().zip(gw.chunks_exact_mut(k)) {
            for (wp, &hp) in wrow.iter_mut().zip(hrow) {
                *wp += v * hp;
            }
        }
    }
}

fn tanh_inplace(blas: bool, x: &mut [f32]) {
    if blas {
        let n = x.len() as i32;
        unsafe { vvtanhf(x.as_mut_ptr(), x.as_ptr(), &n) };
        return;
    }
    for v in x.iter_mut() {
        *v = v.tanh();
    }
}

// ------------------------------------------------------------------ network
#[derive(Clone)]
struct Net {
    nin: [usize; LAYERS],
    nout: [usize; LAYERS],
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    wt: Vec<Vec<f32>>,
    gw: Vec<Vec<f32>>,
    gb: Vec<Vec<f32>>,
    mw: Vec<Vec<f32>>,
    vw: Vec<Vec<f32>>,
    mb: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
}

impl Net {
    fn new() -> Net {
        let mut nin = [WIDTH; LAYERS];
        let mut nout = [WIDTH; LAYERS];
        nin[0] = 2;
        nout[LAYERS - 1] = 1;
        let mats = |_: ()| (0..LAYERS).map(|l| vec![0.0f32; nin[l] * nout[l]]).collect::<Vec<_>>();
        let vecs = |_: ()| (0..LAYERS).map(|l| vec![0.0f32; nout[l]]).collect::<Vec<_>>();
        Net {
            nin, nout,
            w: mats(()), b: vecs(()), wt: mats(()), gw: mats(()), gb: vecs(()),
            mw: mats(()), vw: mats(()), mb: vecs(()), vb: vecs(()),
        }
    }
    fn transpose(&mut self) {
        for l in 0..LAYERS {
            let (i, o) = (self.nin[l], self.nout[l]);
            for j in 0..o {
                for p in 0..i {
                    self.wt[l][p * o + j] = self.w[l][j * i + p];
                }
            }
        }
    }
    fn zero_grad(&mut self) {
        for l in 0..LAYERS {
            self.gw[l].fill(0.0);
            self.gb[l].fill(0.0);
        }
    }
}

struct Rng {
    s: [u64; 4],
} // xoshiro256++

impl Rng {
    fn new(mut seed: u64) -> Rng {
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *v = z ^ (z >> 31);
        }
        Rng { s }
    }
    fn next(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 * (1.0 / 16777216.0)
    }
}

/// Work arrays. The four streams (value, x-slope, x-curvature, t-slope) of the collocation batch
/// are the row blocks [0,N), [N,2N), [2N,3N), [3N,4N) of one matrix, so each layer is one product.
struct Work {
    hin: Vec<Vec<f32>>,
    a: Vec<Vec<f32>>,
    g: Vec<f32>,
    g2: Vec<f32>,
    vin: Vec<Vec<f32>>,
    va: Vec<Vec<f32>>,
    vg: Vec<f32>,
    vg2: Vec<f32>,
}

impl Work {
    fn new(net: &Net) -> Work {
        Work {
            hin: (0..LAYERS).map(|l| vec![0.0; 4 * N_COL * net.nin[l]]).collect(),
            a: (0..LAYERS).map(|l| vec![0.0; 4 * N_COL * net.nout[l]]).collect(),
            g: vec![0.0; 4 * N_COL * WIDTH],
            g2: vec![0.0; 4 * N_COL * WIDTH],
            vin: (0..LAYERS).map(|l| vec![0.0; N_VAL * net.nin[l]]).collect(),
            va: (0..LAYERS).map(|l| vec![0.0; N_VAL * net.nout[l]]).collect(),
            vg: vec![0.0; N_VAL * WIDTH],
            vg2: vec![0.0; N_VAL * WIDTH],
        }
    }
}

/// Residual loss mean(r^2) on the collocation batch `xt` (N x 2); accumulates the weight gradient.
fn residual_pass(blas: bool, net: &mut Net, w: &mut Work, xt: &[f32]) -> f32 {
    let n = N_COL;
    {
        let h0 = &mut w.hin[0];
        for i in 0..n {
            h0[2 * i] = xt[2 * i];
            h0[2 * i + 1] = xt[2 * i + 1];
            h0[2 * (n + i)] = 1.0;
            h0[2 * (n + i) + 1] = 0.0;
            h0[2 * (2 * n + i)] = 0.0;
            h0[2 * (2 * n + i) + 1] = 0.0;
            h0[2 * (3 * n + i)] = 0.0;
            h0[2 * (3 * n + i) + 1] = 1.0;
        }
    }
    for l in 0..LAYERS {
        let (nin, nout) = (net.nin[l], net.nout[l]);
        mul_bt(blas, &w.hin[l], &net.w[l], &net.wt[l], &mut w.a[l], 4 * n, nin, nout);
        for row in w.a[l][..n * nout].chunks_exact_mut(nout) {
            for (v, &b) in row.iter_mut().zip(&net.b[l]) {
                *v += b;
            }
        }
        if l == LAYERS - 1 {
            break;
        }
        let m = n * nout;
        tanh_inplace(blas, &mut w.a[l][..m]);
        let (s, rest) = w.a[l].split_at(m);
        let (ax, rest) = rest.split_at(m);
        let (axx, at) = rest.split_at(m);
        let (h, rest) = w.hin[l + 1].split_at_mut(m);
        let (hx, rest) = rest.split_at_mut(m);
        let (hxx, ht) = rest.split_at_mut(m);
        for i in 0..m {
            let d1 = 1.0 - s[i] * s[i];
            let d2 = -2.0 * s[i] * d1;
            h[i] = s[i];
            hx[i] = d1 * ax[i];
            hxx[i] = d2 * ax[i] * ax[i] + d1 * axx[i];
            ht[i] = d1 * at[i];
        }
    }
    let mut loss = 0.0f64;
    {
        let out = &w.a[LAYERS - 1];
        let g = &mut w.g;
        for i in 0..n {
            let r = out[3 * n + i] - ALPHA * out[2 * n + i];
            loss += (r as f64) * (r as f64);
            let gr = 2.0 * r / n as f32;
            g[i] = 0.0;
            g[n + i] = 0.0;
            g[2 * n + i] = -ALPHA * gr;
            g[3 * n + i] = gr;
        }
    }
    let mut gcur = std::mem::take(&mut w.g);
    let mut gnext = std::mem::take(&mut w.g2);
    for l in (0..LAYERS).rev() {
        let (nin, nout) = (net.nin[l], net.nout[l]);
        add_gt_h(blas, &gcur, &w.hin[l], &mut net.gw[l], 4 * n, nout, nin);
        for row in gcur[..n * nout].chunks_exact(nout) {
            for (gb, &v) in net.gb[l].iter_mut().zip(row) {
                *gb += v;
            }
        }
        if l == 0 {
            break;
        }
        mul_w(blas, &gcur, &net.w[l], &mut gnext, 4 * n, nout, nin);
        let m = n * nin;
        let (s, rest) = w.a[l - 1].split_at(m);
        let (ax, rest) = rest.split_at(m);
        let (axx, at) = rest.split_at(m);
        let (gh, rest) = gnext.split_at_mut(m);
        let (ghx, rest) = rest.split_at_mut(m);
        let (ghxx, ght) = rest.split_at_mut(m);
        for i in 0..m {
            let d1 = 1.0 - s[i] * s[i];
            let d2 = -2.0 * s[i] * d1;
            let d3 = -2.0 * d1 * (1.0 - 3.0 * s[i] * s[i]);
            let a = gh[i] * d1
                + ghx[i] * d2 * ax[i]
                + ghxx[i] * (d3 * ax[i] * ax[i] + d2 * axx[i])
                + ght[i] * d2 * at[i];
            let bx = ghx[i] * d1 + ghxx[i] * 2.0 * d2 * ax[i];
            gh[i] = a;
            ghx[i] = bx;
            ghxx[i] *= d1;
            ght[i] *= d1;
        }
        std::mem::swap(&mut gcur, &mut gnext);
    }
    w.g = gcur;
    w.g2 = gnext;
    (loss / n as f64) as f32
}

/// sum_i coef_i/2 (u(x_i, t_i) - target_i)^2 on the value batch (rows: x, t, target, coef).
fn value_pass(blas: bool, net: &mut Net, w: &mut Work, batch: &[f32]) -> f32 {
    let n = N_VAL;
    for i in 0..n {
        w.vin[0][2 * i] = batch[4 * i];
        w.vin[0][2 * i + 1] = batch[4 * i + 1];
    }
    for l in 0..LAYERS {
        let (nin, nout) = (net.nin[l], net.nout[l]);
        mul_bt(blas, &w.vin[l], &net.w[l], &net.wt[l], &mut w.va[l], n, nin, nout);
        for row in w.va[l].chunks_exact_mut(nout) {
            for (v, &b) in row.iter_mut().zip(&net.b[l]) {
                *v += b;
            }
        }
        if l == LAYERS - 1 {
            break;
        }
        tanh_inplace(blas, &mut w.va[l]);
        let (va, vin) = (&w.va[l], &mut w.vin[l + 1]);
        vin.copy_from_slice(va);
    }
    let mut gcur = std::mem::take(&mut w.vg);
    let mut gnext = std::mem::take(&mut w.vg2);
    let mut loss = 0.0f64;
    for i in 0..n {
        let e = w.va[LAYERS - 1][i] - batch[4 * i + 2];
        let c = batch[4 * i + 3];
        loss += 0.5 * (c as f64) * (e as f64) * (e as f64);
        gcur[i] = c * e;
    }
    for l in (0..LAYERS).rev() {
        let (nin, nout) = (net.nin[l], net.nout[l]);
        add_gt_h(blas, &gcur, &w.vin[l], &mut net.gw[l], n, nout, nin);
        for row in gcur[..n * nout].chunks_exact(nout) {
            for (gb, &v) in net.gb[l].iter_mut().zip(row) {
                *gb += v;
            }
        }
        if l == 0 {
            break;
        }
        mul_w(blas, &gcur, &net.w[l], &mut gnext, n, nout, nin);
        for (g, &s) in gnext[..n * nin].iter_mut().zip(&w.va[l - 1]) {
            *g *= 1.0 - s * s;
        }
        std::mem::swap(&mut gcur, &mut gnext);
    }
    w.vg = gcur;
    w.vg2 = gnext;
    loss as f32
}

fn adam(net: &mut Net, step: i32) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let c1 = 1.0 - b1.powf(step as f32);
    let c2 = 1.0 - b2.powf(step as f32);
    let update = |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
        for i in 0..p.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= LR * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
        }
    };
    for l in 0..LAYERS {
        update(&mut net.w[l], &net.gw[l], &mut net.mw[l], &mut net.vw[l]);
        update(&mut net.b[l], &net.gb[l], &mut net.mb[l], &mut net.vb[l]);
    }
}

fn sample(rng: &mut Rng, xt: &mut [f32], val: &mut [f32]) {
    for v in xt.iter_mut() {
        *v = rng.uniform();
    }
    for i in 0..N_IC {
        let x = rng.uniform();
        val[4 * i] = x;
        val[4 * i + 1] = 0.0;
        val[4 * i + 2] = (PI * x).sin();
        val[4 * i + 3] = 2.0 * W_IC / N_IC as f32;
    }
    for i in 0..N_BC {
        let v = &mut val[4 * (N_IC + i)..4 * (N_IC + i) + 4];
        v[0] = (i % 2) as f32;
        v[1] = rng.uniform();
        v[2] = 0.0;
        v[3] = 2.0 * W_BC / N_BC as f32;
    }
}

fn rel_l2(blas: bool, net: &Net, w: &mut Work) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let mut batch = vec![0.0f32; 4 * N_VAL];
    let total = 101 * 101;
    let mut start = 0;
    while start < total {
        for i in 0..N_VAL {
            let idx = (start + i).min(total - 1);
            batch[4 * i] = (idx / 101) as f32 / 100.0;
            batch[4 * i + 1] = (idx % 101) as f32 / 100.0;
        }
        let mut scratch = net.clone();
        value_pass(blas, &mut scratch, w, &batch);
        for i in 0..N_VAL {
            if start + i >= total {
                break;
            }
            let (x, t) = (batch[4 * i] as f64, batch[4 * i + 1] as f64);
            let exact = (-(ALPHA as f64) * (PI as f64) * (PI as f64) * t).exp() * ((PI as f64) * x).sin();
            let u = w.va[LAYERS - 1][i] as f64;
            num += (u - exact) * (u - exact);
            den += exact * exact;
        }
        start += N_VAL;
    }
    (num / den).sqrt()
}

fn read_f32(f: &mut File, out: &mut [f32]) {
    let mut buf = vec![0u8; 4 * out.len()];
    f.read_exact(&mut buf).expect("short check input");
    for (v, c) in out.iter_mut().zip(buf.chunks_exact(4)) {
        *v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
}

fn write_f32(f: &mut File, data: &[f32]) {
    let mut buf = Vec::with_capacity(4 * data.len());
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mut steps, mut seed, mut mode) = (3000usize, 0u64, String::from("blas"));
    let mut check: Option<(String, String)> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { steps = args[i + 1].parse().unwrap(); i += 1; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 1; }
            "--mode" => { mode = args[i + 1].clone(); i += 1; }
            "--check" => { check = Some((args[i + 1].clone(), args[i + 2].clone())); i += 2; }
            _ => {}
        }
        i += 1;
    }
    let blas = mode == "blas";
    let mut net = Net::new();
    let mut work = Work::new(&net);
    let mut xt = vec![0.0f32; 2 * N_COL];
    let mut val = vec![0.0f32; 4 * N_VAL];

    if let Some((fin, fout)) = check {
        let mut f = File::open(fin).expect("check input");
        for l in 0..LAYERS {
            read_f32(&mut f, &mut net.w[l]);
            read_f32(&mut f, &mut net.b[l]);
        }
        read_f32(&mut f, &mut xt);
        read_f32(&mut f, &mut val);
        net.transpose();
        net.zero_grad();
        let loss = residual_pass(blas, &mut net, &mut work, &xt) + value_pass(blas, &mut net, &mut work, &val);
        let mut o = File::create(fout).unwrap();
        write_f32(&mut o, &[loss]);
        for l in 0..LAYERS {
            write_f32(&mut o, &net.gw[l]);
            write_f32(&mut o, &net.gb[l]);
        }
        return;
    }

    let mut init = Rng::new(seed);
    let mut rng = Rng::new(1000 + seed);
    for l in 0..LAYERS {
        let bound = (6.0f32 / (net.nin[l] + net.nout[l]) as f32).sqrt();
        for v in net.w[l].iter_mut() {
            *v = (2.0 * init.uniform() - 1.0) * bound;
        }
    }
    let (mut t_sample, mut t_resid, mut t_value, mut t_adam) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut loss = 0.0f32;
    let tic = Instant::now();
    for step in 1..=steps {
        let t0 = Instant::now();
        sample(&mut rng, &mut xt, &mut val);
        let t1 = Instant::now();
        net.transpose();
        net.zero_grad();
        loss = residual_pass(blas, &mut net, &mut work, &xt);
        let t2 = Instant::now();
        loss += value_pass(blas, &mut net, &mut work, &val);
        let t3 = Instant::now();
        adam(&mut net, step as i32);
        let t4 = Instant::now();
        t_sample += (t1 - t0).as_secs_f64();
        t_resid += (t2 - t1).as_secs_f64();
        t_value += (t3 - t2).as_secs_f64();
        t_adam += (t4 - t3).as_secs_f64();
    }
    let elapsed = tic.elapsed().as_secs_f64();
    net.transpose();
    let k = 1e3 / steps as f64;
    println!(
        "{{\"impl\": \"rust-{}\", \"seed\": {}, \"steps\": {}, \"ms_per_step\": {:.4}, \"rel_l2\": {:.6e}, \"loss\": {:.6e}, \
         \"stages_ms\": {{\"sample\": {:.4}, \"residual forward and backward\": {:.4}, \"initial and boundary\": {:.4}, \
         \"optimizer\": {:.4}}}}}",
        mode, seed, steps, elapsed * k, rel_l2(blas, &net, &mut work), loss as f64,
        t_sample * k, t_resid * k, t_value * k, t_adam * k
    );
}
