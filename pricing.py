"""
pricing.py

The analysis and pricing logic:
  - price elasticity estimated by regression (volume vs price relative to competitors)
  - profit maximizing price recommendation inside competitive guardrails
  - compliance audit of current site prices against the recommended strategy
  - cost and competitor alerts for market and tax changes
  - administered price changes with a change log
"""
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

GUARDRAIL_LOW = -3.0    # lowest allowed gap to competitor, cents per litre
GUARDRAIL_HIGH = 2.0    # highest allowed gap to competitor, cents per litre
AUDIT_TOLERANCE = 1.0   # allowed distance from the recommended price, cents


def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM daily_sales ORDER BY date, site_id", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_elasticity(df):
    """Log log regression: ln(volume) on ln(own price / competitor price),
    controlling for day of week and site. Returns the price ratio coefficient."""
    d = df.copy()
    d["ln_vol"] = np.log(d["volume"])
    d["ln_ratio"] = np.log(d["own_price"] / d["competitor_price"])
    dow = pd.get_dummies(d["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    site = pd.get_dummies(d["site_id"], prefix="site", drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), d["ln_ratio"], dow, site], axis=1)
    y = d["ln_vol"].values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    pred = X.values @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"elasticity": float(beta[1]), "r2": float(r2), "observations": int(len(d))}


def recommend_price(df, site_id, elasticity, guardrail_low=GUARDRAIL_LOW, guardrail_high=GUARDRAIL_HIGH):
    """Searches prices between the guardrails and picks the one with the highest
    expected daily gross profit under the fitted demand curve."""
    s = df[df["site_id"] == site_id].sort_values("date")
    latest = s.iloc[-1]
    cost, competitor = latest["cost"], latest["competitor_price"]

    recent = s.tail(14)
    parity_volume = float((recent["volume"] / (recent["own_price"] / recent["competitor_price"]) ** elasticity).mean())

    def profit(price):
        volume = parity_volume * (price / competitor) ** elasticity
        return (price - cost) / 100 * volume, volume

    gaps = np.arange(guardrail_low, guardrail_high + 0.05, 0.1)
    best = max(gaps, key=lambda g: profit(competitor + g)[0])
    best_price = round(competitor + float(best), 1)
    best_profit, best_volume = profit(best_price)
    current_profit, current_volume = profit(latest["own_price"])

    return {
        "site_id": site_id,
        "competitor_price": float(competitor),
        "cost": float(cost),
        "current_price": float(latest["own_price"]),
        "recommended_price": best_price,
        "gap_to_competitor": round(best_price - float(competitor), 1),
        "expected_daily_volume": round(best_volume),
        "expected_daily_profit": round(best_profit, 2),
        "current_daily_profit": round(current_profit, 2),
        "profit_change_pct": round((best_profit / current_profit - 1) * 100, 1),
    }


def recommend_all(df, elasticity):
    return pd.DataFrame([recommend_price(df, s, elasticity) for s in sorted(df["site_id"].unique())])


def _current_prices(db_path):
    conn = sqlite3.connect(db_path)
    cp = pd.read_sql("SELECT site_id, price FROM current_prices", conn)
    conn.close()
    return dict(zip(cp["site_id"], cp["price"]))


def compliance_audit(db_path, df, elasticity, tolerance=AUDIT_TOLERANCE):
    """Compares the price each site is actually posting to the recommended strategy price."""
    current = _current_prices(db_path)
    recs = recommend_all(df, elasticity)
    rows = []
    for r in recs.itertuples():
        posted = current[r.site_id]
        gap = round(posted - r.competitor_price, 1)
        deviation = round(posted - r.recommended_price, 1)
        if gap < GUARDRAIL_LOW or gap > GUARDRAIL_HIGH:
            status = "Guardrail Breach"
        elif abs(deviation) > tolerance:
            status = "Off Strategy"
        else:
            status = "Compliant"
        rows.append({"site_id": r.site_id, "posted_price": posted, "recommended_price": r.recommended_price,
                     "deviation": deviation, "gap_to_competitor": gap, "status": status})
    return pd.DataFrame(rows)


def cost_alerts(df, cost_threshold=2.0, competitor_threshold=4.0):
    """Flags day over day jumps in cost (2 cents or more) or competitor price (4 cents or more),
    the signature of a tax adjustment or a large market move. The competitor threshold is
    higher because competitor prices carry normal day to day noise."""
    d = df.sort_values(["site_id", "date"]).copy()
    d["cost_change"] = d.groupby("site_id")["cost"].diff()
    d["competitor_change"] = d.groupby("site_id")["competitor_price"].diff()
    hits = d[(d["cost_change"].abs() >= cost_threshold) | (d["competitor_change"].abs() >= competitor_threshold)]
    return hits[["date", "site_id", "cost_change", "competitor_change"]].reset_index(drop=True)


def change_price(db_path, site_id, new_price, reason, competitor_price):
    """Administers a site price change. Rejects changes outside the guardrails
    and records every accepted change in the change log."""
    gap = new_price - competitor_price
    if gap < GUARDRAIL_LOW or gap > GUARDRAIL_HIGH:
        raise ValueError(
            f"Rejected: {new_price} is {gap:+.1f} cents from competitor, outside the "
            f"{GUARDRAIL_LOW} to {GUARDRAIL_HIGH} guardrail."
        )
    conn = sqlite3.connect(db_path)
    old = conn.execute("SELECT price FROM current_prices WHERE site_id = ?", (site_id,)).fetchone()[0]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("UPDATE current_prices SET price = ?, updated_at = ? WHERE site_id = ?", (new_price, ts, site_id))
    conn.execute(
        "INSERT INTO price_change_log (site_id, old_price, new_price, reason, changed_at) VALUES (?, ?, ?, ?, ?)",
        (site_id, old, new_price, reason, ts),
    )
    conn.commit()
    conn.close()
    return {"site_id": site_id, "old_price": old, "new_price": new_price}


def get_change_log(db_path):
    conn = sqlite3.connect(db_path)
    log = pd.read_sql("SELECT * FROM price_change_log ORDER BY id", conn)
    conn.close()
    return log
