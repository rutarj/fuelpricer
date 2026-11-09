"""
test_fuelpricer.py

Run with: python -m pytest test_fuelpricer.py
"""
import os
import tempfile

import pytest

import data_gen
import pricing
import reports


@pytest.fixture()
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    data_gen.build_database(path)
    yield path
    os.remove(path)


def test_elasticity_recovers_true_value(db):
    fit = pricing.fit_elasticity(pricing.load_data(db))
    assert -7.0 < fit["elasticity"] < -5.0
    assert fit["r2"] > 0.9


def test_recommendations_stay_inside_guardrails(db):
    df = pricing.load_data(db)
    recs = pricing.recommend_all(df, pricing.fit_elasticity(df)["elasticity"])
    assert len(recs) == 8
    assert recs["gap_to_competitor"].between(pricing.GUARDRAIL_LOW - 0.05, pricing.GUARDRAIL_HIGH + 0.05).all()


def test_audit_flags_guardrail_breach_and_off_strategy(db):
    df = pricing.load_data(db)
    e = pricing.fit_elasticity(df)["elasticity"]
    audit = pricing.compliance_audit(db, df, e)
    statuses = set(audit["status"])
    assert "Guardrail Breach" in statuses
    assert "Off Strategy" in statuses


def test_price_change_within_guardrail_is_logged(db):
    df = pricing.load_data(db)
    comp = df[df["site_id"] == "Site 03"].sort_values("date").iloc[-1]["competitor_price"]
    pricing.change_price(db, "Site 03", round(comp + 1.0, 1), "Align to strategy", comp)
    log = pricing.get_change_log(db)
    assert len(log) == 1
    assert log.iloc[0]["site_id"] == "Site 03"


def test_price_change_outside_guardrail_is_rejected(db):
    df = pricing.load_data(db)
    comp = df[df["site_id"] == "Site 03"].sort_values("date").iloc[-1]["competitor_price"]
    with pytest.raises(ValueError):
        pricing.change_price(db, "Site 03", comp + 6.0, "Too high", comp)
    assert pricing.get_change_log(db).empty


def test_change_brings_site_into_compliance(db):
    df = pricing.load_data(db)
    e = pricing.fit_elasticity(df)["elasticity"]
    rec = pricing.recommend_price(df, "Site 03", e)
    pricing.change_price(db, "Site 03", rec["recommended_price"], "Apply recommendation", rec["competitor_price"])
    audit = pricing.compliance_audit(db, df, e)
    assert audit[audit["site_id"] == "Site 03"].iloc[0]["status"] == "Compliant"


def test_tax_adjustment_triggers_cost_alert(db):
    df = pricing.load_data(db)
    alerts = pricing.cost_alerts(df)
    assert not alerts.empty
    assert (alerts["date"] == "2026-08-28").all() or (alerts["date"].astype(str).str.startswith("2026-08-28")).any()
    assert alerts["site_id"].nunique() == 8


def test_weekly_report_covers_all_sites_with_volume_change(db):
    df = pricing.load_data(db)
    weekly = reports.weekly_report(df)
    assert len(weekly) == 8
    assert "volume_change_pct" in weekly.columns


def test_export_writes_excel_and_csv(db, tmp_path):
    df = pricing.load_data(db)
    e = pricing.fit_elasticity(df)["elasticity"]
    path = reports.export_all(df, pricing.recommend_all(df, e), pricing.compliance_audit(db, df, e), out_dir=str(tmp_path))
    assert os.path.exists(path)
    assert (tmp_path / "weekly.csv").exists()
