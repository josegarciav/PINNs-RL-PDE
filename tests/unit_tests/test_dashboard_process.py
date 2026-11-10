"""The dashboard's Process view: the fouling monitor on a historian export."""

import pinnrl.dashboard as dashboard


def test_process_view_runs_on_the_demo_export(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "RESULTS_DIR", str(tmp_path))
    export, tags = dashboard.write_process_demo(1)
    assert export.endswith("evaporator_export.csv") and tags.endswith("evaporator_tags.yaml")
    summary, fouling, records = dashboard.update_process_view(1, export, tags, 3e-4, 0.3, 11)
    assert "Cleaning threshold" in summary and "Steam record offset" in summary
    names = [trace.name for trace in fouling.data]
    assert "fouling, filtered" in names and "fouling, in hindsight" in names
    assert any(name.startswith("time of cleaning") for name in names)
    assert len(records.data) == 4  # a record and a prediction per sensor
    # Without a threshold or a steam uncertainty there is no prognosis and no offset.
    summary, fouling, _ = dashboard.update_process_view(1, export, tags, None, 0, 11)
    assert "Cleaning threshold" not in summary and "Steam record offset" not in summary
    assert not any((trace.name or "").startswith("time of cleaning") for trace in fouling.data)


def test_process_view_reports_a_bad_input_instead_of_raising(tmp_path):
    summary, fouling, records = dashboard.update_process_view(None, None, None, 3e-4, 0, 51)
    assert summary == "" and not fouling.data and not records.data
    summary, _, _ = dashboard.update_process_view(1, "", "", 3e-4, 0, 51)
    assert "both needed" in summary
    tags = tmp_path / "tags.yaml"
    tags.write_text("inputs: {}\n")
    summary, fouling, _ = dashboard.update_process_view(
        1, str(tmp_path / "missing.csv"), str(tags), 3e-4, 0, 51
    )
    assert "missing.csv" in summary and not fouling.data


def test_process_view_passes_the_rate_walk_to_the_monitor(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "RESULTS_DIR", str(tmp_path))
    export, tags = dashboard.write_process_demo(1)
    tight, _, _ = dashboard.update_process_view(1, export, tags, 3e-4, 0, 11, 1e-10)
    loose, _, _ = dashboard.update_process_view(1, export, tags, 3e-4, 0, 11, 1e-8)

    def rate_std(text):  # "Rate: 6.6e-04 ± 6.8e-05 m²K/W per hour."
        return float(text.split("Rate:")[1].split("±")[1].split()[0])

    assert rate_std(loose) > 3 * rate_std(tight)
