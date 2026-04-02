from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from real_data_pipeline import COINS, backfill_coin_history, bootstrap_placeholder_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill real daily OHLCV history into CSV storage.")
    parser.add_argument("--coin", choices=COINS + ["all"], default="all")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD override for the selected coin(s).")
    args = parser.parse_args()

    bootstrap_placeholder_files()
    coins = COINS if args.coin == "all" else [args.coin]
    for coin in coins:
        result = backfill_coin_history(coin, start_date=args.start_date)
        print(json_dumps(result))


def json_dumps(payload) -> str:
    import json
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    main()
