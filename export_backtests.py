from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from real_data_pipeline import BACKTEST_SELECTED_CSV, bootstrap_placeholder_files, export_selected_backtests


def main() -> None:
    bootstrap_placeholder_files()
    df = export_selected_backtests()
    print(f"selected_backtests_rows={len(df)}")
    print(f"selected_backtests_csv={BACKTEST_SELECTED_CSV}")


if __name__ == "__main__":
    main()
