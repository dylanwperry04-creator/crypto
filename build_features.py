from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from real_data_pipeline import FEATURE_MATRIX_CSV, DATA_QUALITY_JSON, build_feature_matrix, bootstrap_placeholder_files


def main() -> None:
    bootstrap_placeholder_files()
    df = build_feature_matrix()
    print(f"feature_matrix_rows={len(df)}")
    print(f"feature_matrix_csv={FEATURE_MATRIX_CSV}")
    print(f"data_quality_json={DATA_QUALITY_JSON}")


if __name__ == "__main__":
    main()
