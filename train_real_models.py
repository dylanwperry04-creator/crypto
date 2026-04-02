from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from real_data_pipeline import BACKTEST_PREDICTIONS_CSV, BEST_MODEL_METADATA_PATH, MODEL_RESULTS_CSV, bootstrap_placeholder_files, train_compare_models


def main() -> None:
    bootstrap_placeholder_files()
    outcomes = train_compare_models()
    for outcome in outcomes:
        print(outcome)
    print(f"model_results_csv={MODEL_RESULTS_CSV}")
    print(f"backtest_predictions_csv={BACKTEST_PREDICTIONS_CSV}")
    print(f"best_model_metadata_json={BEST_MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
