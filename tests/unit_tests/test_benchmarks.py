"""Smoke tests for the ``pinnrl.benchmarks`` subpackage and CLI."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pinnrl.benchmarks import (
    SUPPORTED_STRATEGIES,
    run_sampling_benchmark,
    solve_heat_1d,
    solve_wave_1d,
)
from pinnrl.benchmarks.cli import build_parser, main


class TestFDM:
    def test_heat_solver_matches_analytical(self):
        result = solve_heat_1d(nt=2001, nx=51, t_max=0.5)
        assert result.l2_error < 1e-3
        assert result.u.shape == (2001, 51)
        assert result.u_exact_final.shape == (51,)
        assert result.wall_time_s >= 0.0

    def test_heat_solver_rejects_unstable_grid(self):
        with pytest.raises(ValueError, match="unstable"):
            solve_heat_1d(nt=200, nx=201, alpha=0.1)

    def test_wave_solver_matches_analytical(self):
        result = solve_wave_1d(nt=2001, nx=101, t_max=0.5)
        assert result.l2_error < 1e-1
        assert result.u.shape == (2001, 101)

    def test_wave_solver_rejects_unstable_grid(self):
        with pytest.raises(ValueError, match="CFL"):
            solve_wave_1d(nt=200, nx=2001, c=1.0)


class TestSamplingBenchmark:
    @pytest.mark.parametrize("pde", ["heat", "wave"])
    def test_runs_all_strategies(self, pde):
        results = run_sampling_benchmark(
            pde, strategies=SUPPORTED_STRATEGIES, epochs=5, batch_size=64
        )
        assert [r.strategy for r in results] == list(SUPPORTED_STRATEGIES)
        for r in results:
            assert r.epochs == 5
            assert r.l2_error >= 0.0
            assert len(r.history) == 5

    def test_rejects_unknown_pde(self):
        with pytest.raises(ValueError, match="Unknown pde_name"):
            run_sampling_benchmark("schrodinger", strategies=["uniform"], epochs=2)

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_sampling_benchmark("heat", strategies=["mystery"], epochs=2)

    def test_seed_reproducibility(self):
        a = run_sampling_benchmark("heat", strategies=["uniform"], epochs=5, seed=42)[0]
        b = run_sampling_benchmark("heat", strategies=["uniform"], epochs=5, seed=42)[0]
        assert a.l2_error == pytest.approx(b.l2_error, rel=1e-6)


class TestCLI:
    def test_parser_lists_subcommands(self):
        parser = build_parser()
        ns = parser.parse_args(["fdm", "--pde", "heat"])
        assert ns.cmd == "fdm"
        assert ns.pde == "heat"

    def test_fdm_subcommand_writes_csv(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "out.csv"
        rc = main(["fdm", "--pde", "heat", "--nx", "51", "--nt", "1001", "--csv", str(csv_path)])
        assert rc == 0
        assert csv_path.exists()
        with csv_path.open() as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["pde", "params", "nx", "nt", "l2_error", "max_error", "wall_time_s"]
        assert rows[1][0] == "heat"

    def test_sampling_subcommand_writes_csv(self, tmp_path: Path, capsys):
        csv_path = tmp_path / "samp.csv"
        rc = main(
            [
                "sampling",
                "--pde",
                "heat",
                "--strategies",
                "uniform",
                "stratified",
                "--epochs",
                "5",
                "--batch-size",
                "64",
                "--csv",
                str(csv_path),
            ]
        )
        assert rc == 0
        with csv_path.open() as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == [
            "pde",
            "strategy",
            "epochs",
            "l2_error",
            "max_error",
            "final_loss",
            "wall_time_s",
        ]
        assert {row[1] for row in rows[1:]} == {"uniform", "stratified"}
