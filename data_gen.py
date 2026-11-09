"""
data_gen.py

Generates synthetic daily retail fuel data for 8 sites over 90 days and loads
it into SQLite. Prices are in cents per litre, volumes in litres.

The data has a known demand elasticity built in (TRUE_ELASTICITY) so the
analysis can be checked: the regression should recover a value close to it.
A tax adjustment is injected on day 60, raising cost and competitor prices,
so the alert logic has a real regulatory change to catch.

All data is synthetic. It is not Circle K data or any real market data.
Swap in real data (for example weekly retail prices published by Natural
Resources Canada) by loading it into the same daily_sales table.
"""
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "fuelpricer.db"
SITES = [f"Site {i:02d}" for i in range(1, 9)]
TRUE_ELASTICITY = 6.0      # volume falls about 6 percent for a 1 percent price rise vs competitors
START = "2026-06-29"
DAYS = 90
TAX_DAY = 60
TAX_CHANGE = 5.0           # cents per litre added to cost and competitor prices
DOW_FACTOR = [0.95, 1.0, 1.0, 1.0, 1.1, 1.15, 0.9]

SCHEMA = """
DROP TABLE IF EXISTS daily_sales;
DROP TABLE IF EXISTS current_prices;
DROP TABLE IF EXISTS price_change_log;
DROP TABLE IF EXISTS events;

CREATE TABLE daily_sales (
    site_id TEXT NOT NULL,
    date TEXT NOT NULL,
    own_price REAL NOT NULL,
    cost REAL NOT NULL,
    competitor_price REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (site_id, date)
);
CREATE TABLE current_prices (
    site_id TEXT PRIMARY KEY,
    price REAL NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE price_change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    old_price REAL NOT NULL,
    new_price REAL NOT NULL,
    reason TEXT NOT NULL,
    changed_at TEXT NOT NULL
);
CREATE TABLE events (
    date TEXT NOT NULL,
    description TEXT NOT NULL
);
"""


def generate_frame(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=DAYS)
    walk = np.cumsum(rng.normal(0, 0.4, DAYS))
    market = 152 + 3 * np.sin(2 * np.pi * np.arange(DAYS) / 30) + walk
    market = market + np.where(np.arange(DAYS) >= TAX_DAY, TAX_CHANGE, 0.0)

    rows = []
    for s, site in enumerate(SITES):
        freight = rng.uniform(0, 3)
        site_offset = rng.normal(0, 0.5)
        base_volume = rng.uniform(5000, 8000)
        for t, d in enumerate(dates):
            competitor = market[t] + site_offset + rng.normal(0, 0.6)
            cost = market[t] - 27 + freight
            gap = float(np.clip(rng.normal(0.3, 1.6), -4, 3.5))
            own = competitor + gap
            volume = (base_volume * DOW_FACTOR[d.dayofweek]
                      * (own / competitor) ** (-TRUE_ELASTICITY)
                      * np.exp(rng.normal(0, 0.04)))
            rows.append((site, d.strftime("%Y-%m-%d"), round(own, 1), round(cost, 1),
                         round(competitor, 1), round(volume, 0)))
    return pd.DataFrame(rows, columns=["site_id", "date", "own_price", "cost", "competitor_price", "volume"])


def build_database(db_path=DB_PATH, seed=42):
    df = generate_frame(seed)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    df.to_sql("daily_sales", conn, if_exists="append", index=False)

    last = df[df["date"] == df["date"].max()]
    conn.executemany(
        "INSERT INTO current_prices (site_id, price, updated_at) VALUES (?, ?, ?)",
        [(r.site_id, r.own_price, r.date) for r in last.itertuples()],
    )
    tax_date = (pd.Timestamp(START) + pd.Timedelta(days=TAX_DAY)).strftime("%Y-%m-%d")
    conn.execute("INSERT INTO events VALUES (?, ?)", (tax_date, "Simulated tax adjustment: +5.0 cents per litre"))
    conn.commit()
    conn.close()
    return len(df)


if __name__ == "__main__":
    n = build_database()
    print(f"Loaded {n} daily site records into {DB_PATH}.")
