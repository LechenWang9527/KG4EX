#!/usr/bin/env python3
"""
Build offline evaluation statistics for KG4EX.

The script estimates each exercise's objective correctness from a pyKT output
CSV and Q matrix. The output pickle is used by ACC and ZPD-EMCC evaluation.
"""
"""
sh:
python build_stats.py \
  --csv_path ../../pyKT_example/output_final_akt_fpkc_h5_g09.csv \
  --q_file Q.txt
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("questions", "concepts", "responses")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build offline evaluation statistics for KG4EX."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the pyKT output CSV file.",
    )
    parser.add_argument(
        "--q_file",
        type=str,
        default="Q.txt",
        help="Path to Q.txt. Relative paths are resolved from this script directory.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="raw_stats_akt.pkl",
        help="Output pickle name. The file is saved in this script directory.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    """Resolve a path relative to the script directory."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def split_sequence(value: Any) -> List[str]:
    """Split one comma-separated sequence while preserving positional alignment."""
    text = str(value)
    if text == "nan":
        text = ""
    return text.split(",")


def load_interactions(csv_path: Path) -> pd.DataFrame:
    """Load pyKT sequences and convert them into interaction-level records."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {missing_columns}")

    records = []
    truncated_rows = 0

    for _, row in df.iterrows():
        questions = split_sequence(row["questions"])
        concepts = split_sequence(row["concepts"])
        responses = split_sequence(row["responses"])

        aligned_length = min(len(questions), len(concepts), len(responses))
        if aligned_length == 0:
            continue

        if aligned_length < max(len(questions), len(concepts), len(responses)):
            truncated_rows += 1

        for idx in range(aligned_length):
            records.append(
                {
                    "question": questions[idx],
                    "concept": concepts[idx],
                    "response": responses[idx],
                }
            )

    if not records:
        raise ValueError("No valid interaction records were found in the CSV file.")

    interaction_df = pd.DataFrame(records)
    interaction_df = interaction_df[interaction_df["concept"].astype(str).str.strip() != ""]
    interaction_df["concept"] = pd.to_numeric(interaction_df["concept"], errors="coerce")
    interaction_df["response"] = pd.to_numeric(interaction_df["response"], errors="coerce")
    interaction_df = interaction_df.dropna(subset=["concept", "response"])

    if interaction_df.empty:
        raise ValueError("No valid interaction records were found after cleaning.")

    interaction_df["concept"] = interaction_df["concept"].astype(int)

    if truncated_rows > 0:
        print(f"检测到 {truncated_rows} 行序列长度不一致，已按最短长度对齐。")

    return interaction_df


def load_q_matrix(q_path: Path) -> List[List[int]]:
    """Load Q.txt as an exercise-concept matrix."""
    if not q_path.exists():
        raise FileNotFoundError(f"Q matrix file does not exist: {q_path}")

    q_matrix = []
    with q_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"Empty row detected in Q matrix at line {line_no}.")
            try:
                q_matrix.append([int(value) for value in text.split(",")])
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer value detected in Q matrix at line {line_no}."
                ) from exc

    if not q_matrix:
        raise ValueError("Q matrix is empty.")

    return q_matrix


def build_exercise_statistics(
    interaction_df: pd.DataFrame,
    q_matrix: List[List[int]],
) -> Dict[str, Any]:
    """Estimate exercise correctness from concept-level historical correctness."""
    concept_correctness = interaction_df.groupby("concept")["response"].mean().to_dict()
    global_correctness = float(interaction_df["response"].mean())

    exercise_df_dict = {}
    for exercise_id, q_row in enumerate(q_matrix):
        concept_ids = [concept_id for concept_id, value in enumerate(q_row) if value == 1]

        if concept_ids:
            concept_values = [
                concept_correctness.get(concept_id, global_correctness)
                for concept_id in concept_ids
            ]
            exercise_correctness = float(np.mean(concept_values))
        else:
            exercise_correctness = global_correctness

        exercise_df_dict[exercise_id] = exercise_correctness

    return {
        "exercise_df_dict": exercise_df_dict,
        "global_df_mean": global_correctness,
    }


def save_statistics(stats_data: Dict[str, Any], output_path: Path) -> None:
    """Save offline statistics as a pickle file."""
    with output_path.open("wb") as file:
        pickle.dump(stats_data, file)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    csv_path = resolve_path(args.csv_path, script_dir)
    q_path = resolve_path(args.q_file, script_dir)
    output_path = script_dir / args.output_name

    print("生成离线评估统计数据。")
    print(f"读取 CSV 文件：{csv_path}")
    print(f"读取 Q 矩阵文件：{q_path}")

    interaction_df = load_interactions(csv_path)
    q_matrix = load_q_matrix(q_path)
    stats_data = build_exercise_statistics(interaction_df, q_matrix)
    save_statistics(stats_data, output_path)

    print("离线评估统计数据生成完成。")
    print(f"有效交互记录数量：{len(interaction_df)}")
    print(f"已生成习题统计数量：{len(stats_data['exercise_df_dict'])}")
    print(f"全局平均正确率：{stats_data['global_df_mean']:.4f}")
    print(f"离线统计文件已保存到：{output_path}")


if __name__ == "__main__":
    main()
