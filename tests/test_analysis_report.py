"""CSV descriptors, SVG histograms, and optional critic QED threshold."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.critic import CriticAgent
from agents.critic_schema import build_evidence_summary
from analysis.molecule_analysis import analyze_molecules
from analysis.report import generate_html_report

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "projects" / "demo_project" / "output" / "sampled-sample.csv"

PASSING_OFFLINE = {
    "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["analyze_molecules"]},
    "steps": {
        "analyze_molecules": {
            "ok": True,
            "total_molecules": 10,
            "duplicate_fraction": 0.0,
            "rdkit": {"available": True, "valid_fraction": 1.0, "qed": {"mean": 0.6, "n": 10}},
        },
        "find_output": {"ok": True, "csv_count": 1},
        "validate_project": {"ok": True},
    },
    "warnings": [],
    "errors": [],
}


def test_analyze_sample_csv_has_sa_pains_and_histograms():
    pytest.importorskip("rdkit")
    result = analyze_molecules(SAMPLE_CSV)
    assert result["ok"] is True
    rdkit = result["rdkit"]
    assert rdkit["available"] is True
    assert rdkit["sa_score"]["available"] is True
    assert rdkit["sa_score"]["mean"] is not None
    assert rdkit["sa_score"]["n"] == rdkit["valid_molecules"]
    assert rdkit["pains"]["available"] is True
    assert rdkit["pains"]["molecules_with_hits"] == 0
    assert rdkit["pains"]["total_hits"] == 0
    assert rdkit["filters"]["qed_pass_threshold"] == 0.5
    assert rdkit["filters"]["qed_pass_count"] == 12
    assert rdkit["filters"]["pains_free_count"] == 15
    histograms = rdkit["histograms"]
    for key in ("qed", "mw", "logp"):
        hist = histograms[key]
        assert hist["n"] == 15
        assert sum(hist["counts"]) == 15
        assert len(hist["bin_edges"]) == len(hist["counts"]) + 1
    assert histograms["qed"]["bin_edges"][0] == 0.0
    assert histograms["qed"]["bin_edges"][-1] == 1.0


def test_analyze_detects_pains_hit(tmp_path: Path):
    pytest.importorskip("rdkit")
    csv_path = tmp_path / "pains.csv"
    csv_path.write_text("SMILES\nCCO\nO=C1C=CC(=O)C=C1\n", encoding="utf-8")
    result = analyze_molecules(csv_path)
    pains = result["rdkit"]["pains"]
    assert pains["available"] is True
    assert pains["molecules_with_hits"] == 1
    assert pains["total_hits"] >= 1
    assert result["rdkit"]["filters"]["pains_free_count"] == 1


def test_analyze_does_not_invent_sa_when_rdkit_missing(monkeypatch):
    import analysis.molecule_analysis as mod

    def _boom() -> tuple:
        raise ImportError("rdkit missing for test")

    monkeypatch.setattr(mod, "_import_rdkit_modules", _boom)
    result = analyze_molecules(SAMPLE_CSV)
    assert result["ok"] is True
    rdkit = result["rdkit"]
    assert rdkit["available"] is False
    assert rdkit["sa_score"]["mean"] is None
    assert rdkit["pains"]["molecules_with_hits"] is None
    assert rdkit["filters"]["qed_pass_count"] is None
    assert rdkit["histograms"]["qed"]["counts"] == []


def test_html_report_embeds_svg_histograms(tmp_path: Path):
    meta = generate_html_report(
        {
            "goal": "charts",
            "project": {},
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True},
            "inventory": {},
            "analysis": {
                "ok": True,
                "csv_path": str(SAMPLE_CSV),
                "analysis_source": "existing_csv",
                "total_molecules": 4,
                "unique_molecules": 4,
                "duplicate_molecules": 0,
                "duplicate_fraction": 0.0,
                "rdkit": {
                    "available": True,
                    "valid_molecules": 4,
                    "invalid_molecules": 0,
                    "valid_fraction": 1.0,
                    "mw": {"mean": 200.0, "stdev": 10.0, "n": 4, "min": 180.0, "max": 220.0},
                    "logp": {"mean": 2.0, "stdev": 0.5, "n": 4, "min": 1.0, "max": 3.0},
                    "qed": {"mean": 0.5, "stdev": 0.1, "n": 4, "min": 0.4, "max": 0.7},
                    "sa_score": {
                        "mean": 2.5,
                        "stdev": 0.2,
                        "n": 4,
                        "min": 2.0,
                        "max": 3.0,
                        "available": True,
                    },
                    "pains": {
                        "available": True,
                        "molecules_with_hits": 1,
                        "total_hits": 1,
                        "hit_fraction": 0.25,
                    },
                    "filters": {
                        "qed_pass_threshold": 0.5,
                        "qed_pass_count": 3,
                        "pains_free_count": 3,
                        "qed_pass_and_pains_free_count": 2,
                    },
                    "histograms": {
                        "qed": {
                            "bin_edges": [0.0, 0.5, 1.0],
                            "counts": [1, 3],
                            "n": 4,
                            "min": 0.2,
                            "max": 0.8,
                        },
                        "mw": {
                            "bin_edges": [100.0, 200.0, 300.0],
                            "counts": [2, 2],
                            "n": 4,
                            "min": 120.0,
                            "max": 280.0,
                        },
                        "logp": {
                            "bin_edges": [0.0, 1.0, 2.0],
                            "counts": [1, 3],
                            "n": 4,
                            "min": 0.1,
                            "max": 1.9,
                        },
                    },
                },
            },
            "critic": {"status": "PASS", "issues": []},
            "warnings": [],
            "errors": [],
        },
        reports_dir=tmp_path,
        filename="histograms.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert html.count("<svg") == 3
    assert 'aria-label="QED distribution histogram"' in html
    assert 'aria-label="Molecular weight distribution histogram"' in html
    assert 'aria-label="logP distribution histogram"' in html
    assert "molecules with PAINS hits" in html
    assert "QED ≥ threshold and PAINS-free" in html
    assert "<h3>Distributions</h3>" in html
    # Charts are first-class, not a JSON-only dump of analysis
    assert "<figure class='chart'>" in html


def test_report_from_sample_csv_includes_charts_and_pains(tmp_path: Path):
    pytest.importorskip("rdkit")
    analysis = analyze_molecules(SAMPLE_CSV)
    analysis["analysis_source"] = "existing_csv"
    meta = generate_html_report(
        {
            "goal": "offline sample",
            "project": {},
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True},
            "inventory": {},
            "analysis": analysis,
            "critic": {"status": "PASS", "issues": []},
            "warnings": [],
            "errors": [],
            "dry_run": True,
            "analysis_source": "existing_csv",
        },
        reports_dir=tmp_path,
        filename="sample_charts.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert html.count("<svg") == 3
    assert "molecules with PAINS hits" in html
    assert ">0<" in html or "0" in html
    assert "SA Score" in html
    assert "Dry-run: REINVENT was not executed" in html


def test_html_report_no_svg_when_histograms_empty(tmp_path: Path):
    meta = generate_html_report(
        {
            "goal": "no rdkit",
            "project": {},
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True},
            "inventory": {},
            "analysis": {
                "ok": True,
                "csv_path": str(SAMPLE_CSV),
                "rdkit": {
                    "available": False,
                    "error": "rdkit missing",
                    "histograms": {
                        "qed": {"bin_edges": [], "counts": [], "n": 0},
                        "mw": {"bin_edges": [], "counts": [], "n": 0},
                        "logp": {"bin_edges": [], "counts": [], "n": 0},
                    },
                    "pains": {"available": False, "molecules_with_hits": None},
                    "filters": {"qed_pass_count": None},
                },
            },
            "critic": {"status": "WARNING", "issues": []},
            "warnings": [],
            "errors": [],
        },
        reports_dir=tmp_path,
        filename="empty_hist.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "<svg" not in html
    assert "No QED values to plot" in html
    assert "PAINS" in html


def test_critic_default_ignores_qed_when_threshold_unset():
    verdict = CriticAgent().review(PASSING_OFFLINE)
    assert verdict["status"] == "PASS"
    assert verdict["thresholds"]["min_mean_qed"] is None


def test_critic_warns_when_mean_qed_below_threshold():
    critic = CriticAgent({"critic": {"min_mean_qed": 0.7}})
    verdict = critic.review(PASSING_OFFLINE)
    assert verdict["status"] == "WARNING"
    assert any("Mean QED" in item and "0.7000" in item for item in verdict["issues"])
    assert verdict["thresholds"]["min_mean_qed"] == 0.7


def test_critic_does_not_invent_qed_when_mean_missing():
    critic = CriticAgent({"critic": {"min_mean_qed": 0.7}})
    results = {
        "plan": {"approve_run": False, "skip_reinvent": True},
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 10,
                "duplicate_fraction": 0.0,
                "rdkit": {"available": True, "valid_fraction": 1.0},
            },
            "find_output": {"ok": True, "csv_count": 1},
            "validate_project": {"ok": True},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = critic.review(results)
    blob = " ".join(verdict["issues"])
    assert "threshold not applied" in blob
    assert "no value invented" in blob
    assert "Mean QED" not in blob
    assert verdict["status"] == "WARNING"


def test_evidence_summary_includes_sa_pains_filters_not_histograms():
    results = {
        "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["analyze_molecules"]},
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "csv_path": "/tmp/sampled.csv",
                "total_molecules": 2,
                "smiles": ["CCO", "c1ccccc1"],
                "rdkit": {
                    "available": True,
                    "valid_fraction": 1.0,
                    "qed": {"mean": 0.5, "n": 2},
                    "sa_score": {"mean": 2.1, "n": 2, "available": True},
                    "pains": {"available": True, "molecules_with_hits": 0, "total_hits": 0},
                    "filters": {"qed_pass_threshold": 0.5, "qed_pass_count": 1},
                    "histograms": {"qed": {"counts": [1, 1], "bin_edges": [0, 0.5, 1]}},
                },
            }
        },
        "errors": [],
        "warnings": [],
    }
    evidence = build_evidence_summary(results, thresholds={"min_mean_qed": 0.3})
    blob = str(evidence)
    assert "CCO" not in blob
    assert evidence["analysis"]["rdkit"]["sa_score"]["mean"] == 2.1
    assert evidence["analysis"]["rdkit"]["pains"]["molecules_with_hits"] == 0
    assert evidence["analysis"]["rdkit"]["filters"]["qed_pass_count"] == 1
    assert "histograms" not in evidence["analysis"]["rdkit"]
    assert evidence["thresholds"]["min_mean_qed"] == 0.3
