"""
main.py: command line interface for FuelPricer.

  python main.py init                       build the synthetic database
  python main.py elasticity                 estimate price elasticity
  python main.py recommend                  recommended price per site
  python main.py audit                      compliance audit of posted prices
  python main.py alerts                     cost and competitor price alerts
  python main.py report daily|weekly|period fuel report by site
  python main.py change SITE PRICE REASON   administer a price change
  python main.py export                     Excel workbook and Power BI CSVs
"""
import argparse

import pandas as pd

import data_gen
import pricing
import reports

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["init", "elasticity", "recommend", "audit", "alerts", "report", "change", "export"])
    p.add_argument("args", nargs="*")
    p.add_argument("--db", default=data_gen.DB_PATH)
    a = p.parse_args()

    if a.command == "init":
        print(f"Loaded {data_gen.build_database(a.db)} daily site records into {a.db}.")
        return

    df = pricing.load_data(a.db)
    fit = pricing.fit_elasticity(df)

    if a.command == "elasticity":
        print(f"Estimated elasticity: {fit['elasticity']:.2f} (R squared {fit['r2']:.3f}, {fit['observations']} observations)")
        print(f"Synthetic data was generated with a true elasticity of -{data_gen.TRUE_ELASTICITY}.")
    elif a.command == "recommend":
        print(pricing.recommend_all(df, fit["elasticity"]).to_string(index=False))
    elif a.command == "audit":
        print(pricing.compliance_audit(a.db, df, fit["elasticity"]).to_string(index=False))
    elif a.command == "alerts":
        print(pricing.cost_alerts(df).to_string(index=False))
    elif a.command == "report":
        kind = a.args[0] if a.args else "weekly"
        print(getattr(reports, f"{kind}_report")(df).to_string(index=False))
    elif a.command == "change":
        site, price, reason = a.args[0], float(a.args[1]), " ".join(a.args[2:]) or "Manual change"
        competitor = df[df["site_id"] == site].sort_values("date").iloc[-1]["competitor_price"]
        print(pricing.change_price(a.db, site, price, reason, competitor))
    elif a.command == "export":
        recs = pricing.recommend_all(df, fit["elasticity"])
        audit = pricing.compliance_audit(a.db, df, fit["elasticity"])
        print("Wrote", reports.export_all(df, recs, audit))


if __name__ == "__main__":
    main()
