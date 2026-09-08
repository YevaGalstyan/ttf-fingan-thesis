"""
Filter chain for the options panel.

Reports the row and contract count after each filter applied in Sec. 4.1.5,
so the counts quoted there can be reproduced.

Run build_options_panel.py first.
"""

import sys
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MASTER_PATH

con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")

q = f"""
SELECT COUNT(*) AS n_rows, COUNT(DISTINCT instrument_id) AS n_contracts
FROM read_parquet('{MASTER_PATH}')
"""

# Filters are cumulative, each adding one condition to the previous.
w1 = "trading_date BETWEEN DATE '2024-09-05' AND DATE '2026-04-24'"  # test split
w2 = w1 + " AND days_to_expiry > 0"                    # no time to maturity at expiry
w3 = w2 + " AND futures_price IS NOT NULL"             # moneyness needs a futures price
w4 = w3 + " AND moneyness_k_over_f BETWEEN 0.5 AND 2.0"  # usable strikes only

steps = [
    ("panel",              None),
    ("test split",         w1),
    ("expiry day dropped", w2),
    ("futures price",      w3),
    ("moneyness band",     w4),
]

for label, where in steps:
    sql = q if where is None else f"{q} WHERE {where}"
    rows, contracts = con.execute(sql).fetchone()
    print(f"{label:20s} {rows:>12,} rows  {contracts:>8,} contracts")