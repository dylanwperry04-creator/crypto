from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from real_data_pipeline import COINS, bootstrap_placeholder_files, incremental_update_all, incremental_update_coin


def main() -> None:
    parser = argparse.ArgumentParser(description="Append only new completed real candles into the local CSV store.")
    parser.add_argument("--coin", choices=COINS + ["all"], default="all")
    args = parser.parse_args()

    bootstrap_placeholder_files()
    if args.coin == "all":
        results = incremental_update_all()
        for result in results:
            print(result)
    else:
        print(incremental_update_coin(args.coin))


if __name__ == "__main__":
    main()
