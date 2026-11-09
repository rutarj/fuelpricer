"""
reports.py

Daily, weekly, and four week period fuel reports by site, plus Excel and CSV
export. The CSV files load directly into Power BI.
"""
import os

import pandas as pd


def _summarize(window):
    g = window.assign(gross_profit=(window["own_price"] - window["cost"]) / 100 * window["volume"])
    out = g.groupby("site_id").agg(
        avg_price=("own_price", "mean"),
        avg_cost=("cost", "mean"),
        avg_competitor=("competitor_price", "mean"),
        total_volume=("volume", "sum"),
        gross_profit=("gross_profit", "sum"),
    ).reset_index()
    out["avg_margin_cpl"] = out["avg_price"] - out["avg_cost"]
    out["avg_gap_to_competitor"] = out["avg_price"] - out["avg_competitor"]
    return out.round(2)


def report(df, days, end_date=None):
    """Summary by site for the last `days` days, with volume change against the prior window."""
    end = pd.Timestamp(end_date) if end_date else df["date"].max()
    start = end - pd.Timedelta(days=days - 1)
    prior_start = start - pd.Timedelta(days=days)

    current = _summarize(df[(df["date"] >= start) & (df["date"] <= end)])
    prior = _summarize(df[(df["date"] >= prior_start) & (df["date"] < start)])

    if not prior.empty:
        merged = current.merge(prior[["site_id", "total_volume"]], on="site_id", suffixes=("", "_prior"))
        merged["volume_change_pct"] = ((merged["total_volume"] / merged["total_volume_prior"] - 1) * 100).round(1)
        current = merged.drop(columns=["total_volume_prior"])
    return current


def daily_report(df):
    return report(df, 1)


def weekly_report(df):
    return report(df, 7)


def period_report(df):
    return report(df, 28)


def export_all(df, recs, audit, out_dir="exports"):
    os.makedirs(out_dir, exist_ok=True)
    sheets = {
        "Daily": daily_report(df),
        "Weekly": weekly_report(df),
        "Period": period_report(df),
        "Recommendations": recs,
        "Compliance Audit": audit,
    }
    xlsx_path = os.path.join(out_dir, "fuel_reports.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    for name, frame in sheets.items():
        frame.to_csv(os.path.join(out_dir, name.lower().replace(" ", "_") + ".csv"), index=False)
    return xlsx_path
