#!/usr/bin/env python3
"""
Unified evaluation script for KG4EX.

This script loads trained embeddings, scores candidate exercises for each user,
and evaluates ACC@N and ZPD-EMCC@N. It supports three scoring backends:
    - cogre
    - triplere
    - transe_adv

The evaluation metrics are kept consistent with the README description:
    - ACC compares model mastery prediction with an IRT-style objective success probability.
    - ZPD-EMCC measures expected learning coverage weighted by PKC and ZPD suitability.
"""
'''
sh:
python test_models.py \
  --model_type cogre \
  --embedding_path ./models/algebra2005/cog_h5_g09
'''

import argparse
import collections
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

csv.field_size_limit(sys.maxsize)


DEFAULT_TOP_N = (10, 20)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate KG4EX recommendation embeddings.")

    parser.add_argument(
        "--model_type",
        choices=("cogre", "triplere", "transe_adv"),
        default="cogre",
        help="Scoring backend used to rank candidate exercises.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/algebra2005",
        help="Dataset directory containing Q.txt, entities.dict, and relations.dict.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="./models/algebra2005/cog_h5_g09",
        help="Directory containing entity_embedding.npy and relation_embedding.npy.",
    )
    parser.add_argument(
        "--test_triples_path",
        type=str,
        default=None,
        help="Path to the test triples file. Defaults to data_path/akt_fpkc_h5_g09_test_triples.txt.",
    )
    parser.add_argument(
        "--offline_stats_path",
        type=str,
        default=None,
        help="Path to raw_stats_akt.pkl. Defaults to data_path/raw_stats_akt.pkl.",
    )
    parser.add_argument(
        "--student_history_path",
        type=str,
        default="../pyKT_example/output_final_akt_fpkc_h5_g09.csv",
        help="Path to the pyKT output CSV used for online student history.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_N),
        help="Top-N values for evaluation.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=12.0,
        help="Margin parameter used in scoring.",
    )
    parser.add_argument(
        "--cogre_alpha",
        type=float,
        default=0.4,
        help="Asymmetric L1 coefficient for CogRE.",
    )
    parser.add_argument(
        "--triplere_u",
        type=float,
        default=1.0,
        help="TripleRE constant u. Use the same value as training.",
    )

    return parser.parse_args()


def resolve_path(path_text: str | None, default_path: Path | None = None) -> Path:
    """Resolve an optional path argument."""
    if path_text is None:
        if default_path is None:
            raise ValueError("Path is not specified.")
        return default_path
    return Path(path_text)


def load_embeddings(embedding_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load entity and relation embeddings from a model directory."""
    try:
        entity_embedding = np.load(embedding_path / "entity_embedding.npy")
        relation_embedding = np.load(embedding_path / "relation_embedding.npy")
    except Exception as exc:
        raise FileNotFoundError(f"Failed to load embeddings from {embedding_path}") from exc

    return entity_embedding, relation_embedding


def load_q_matrix(data_path: Path) -> List[List[int]]:
    """Load Q.txt as a binary exercise-concept matrix."""
    q_matrix: List[List[int]] = []
    q_path = data_path / "Q.txt"

    with q_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"Empty row in Q.txt at line {line_no}.")
            row = [int(value) for value in text.split(",")]
            if any(value not in (0, 1) for value in row):
                raise ValueError(f"Q.txt contains non-binary value at line {line_no}.")
            q_matrix.append(row)

    if not q_matrix:
        raise ValueError("Q.txt is empty.")

    row_lengths = {len(row) for row in q_matrix}
    if len(row_lengths) != 1:
        raise ValueError("Q.txt rows have inconsistent lengths.")

    return q_matrix


def load_id_dict(path: Path) -> Dict[str, int]:
    """Load a KG4EX id dictionary file."""
    id_dict: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                raise ValueError(f"Malformed dictionary line {line_no}: {line.rstrip()}")
            idx, key = parts
            id_dict[key] = int(idx)
    return id_dict


def build_embedding_dicts(
    entity_embedding: np.ndarray,
    relation_embedding: np.ndarray,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build name-to-embedding maps using the id dictionaries."""
    entity_embeddings = {}
    for entity, idx in entity2id.items():
        if idx >= entity_embedding.shape[0]:
            raise IndexError(f"Entity id {idx} for {entity} exceeds entity embedding size.")
        entity_embeddings[entity] = entity_embedding[idx, :]

    relation_embeddings = {}
    for relation, idx in relation2id.items():
        if idx >= relation_embedding.shape[0]:
            raise IndexError(f"Relation id {idx} for {relation} exceeds relation embedding size.")
        relation_embeddings[relation] = relation_embedding[idx, :]

    return entity_embeddings, relation_embeddings


def load_test_triples(test_path: Path) -> Tuple[dict, dict, dict]:
    """Load test triples into user-indexed mlkc, pkc, and exfr dictionaries."""
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test triples file: {test_path}")

    uid_mlkc_dict = {}
    uid_pkc_dict = {}
    uid_exfr_dict = {}

    with test_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed test triple at line {line_no}: {line.rstrip()}")

            item, relation, uid = parts
            uid_mlkc_dict.setdefault(uid, {})
            uid_pkc_dict.setdefault(uid, {})
            uid_exfr_dict.setdefault(uid, {})

            if relation.startswith("mlkc"):
                uid_mlkc_dict[uid][item] = relation
            elif relation.startswith("pkc"):
                uid_pkc_dict[uid][item] = relation
            elif relation.startswith("exfr"):
                uid_exfr_dict[uid][item] = relation

    return uid_mlkc_dict, uid_pkc_dict, uid_exfr_dict


def get_relation_embedding(key: str, relation_embeddings: Dict[str, np.ndarray], default: str = "rec") -> torch.Tensor:
    """Return a relation embedding using the same lookup rule as the old test script."""
    key = str(key).strip()

    if key.endswith(".0") and not key.endswith(".00"):
        key += "0"

    if key in relation_embeddings:
        return torch.from_numpy(relation_embeddings[key])

    if default in relation_embeddings:
        return torch.from_numpy(relation_embeddings[default])

    first_embedding = next(iter(relation_embeddings.values()))
    return torch.zeros_like(torch.from_numpy(first_embedding))

def get_entity_embedding(key: str, entity_embeddings: Dict[str, np.ndarray]) -> torch.Tensor:
    """Return an entity embedding with zero-vector fallback."""
    if key in entity_embeddings:
        return torch.from_numpy(entity_embeddings[key])

    first_embedding = next(iter(entity_embeddings.values()))
    return torch.zeros_like(torch.from_numpy(first_embedding))


def parse_probability_token(token: str, prefix: str, default: float = 0.0) -> float:
    """Parse relation labels such as mlkc0.73 or pkc0.42."""
    try:
        value = float(str(token).replace(prefix, ""))
    except (TypeError, ValueError):
        return default
    return float(np.clip(value, 0.0, 1.0))


def apply_cogre_head(head_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
    """Apply the CogRE head-side relation-gated projection."""
    r_gate_h, _, r_m, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_h = torch.sigmoid(r_gate_h)
    gamma_r = 0.2 * torch.sigmoid(r_res)
    return head_emb * (gate_h + gamma_r) + r_m


def apply_cogre_tail(tail_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
    """Apply the CogRE tail-side relation-gated projection."""
    _, r_gate_t, _, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_t = torch.sigmoid(r_gate_t)
    gamma_r = 0.2 * torch.sigmoid(r_res)
    return tail_emb * (gate_t + gamma_r)


def calc_alpha_l1(diff_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute the asymmetric L1 penalty used by CogRE."""
    abs_diff = torch.abs(diff_tensor)
    penalty = torch.where(diff_tensor > 0, abs_diff, alpha * abs_diff)
    return torch.sum(penalty, dim=-1)


def load_all_exercise_embeddings(q_matrix: List[List[int]], entity_embeddings: Dict[str, np.ndarray]) -> torch.Tensor:
    """Load embeddings for all exercises in Q.txt order."""
    return torch.stack([
        get_entity_embedding(f"ex{exercise_id}", entity_embeddings)
        for exercise_id in range(len(q_matrix))
    ]).float()


def score_candidates_cogre(
    uid_mlkc_dict: dict,
    uid_pkc_dict: dict,
    uid_exfr_dict: dict,
    q_matrix: List[List[int]],
    entity_embeddings: Dict[str, np.ndarray],
    relation_embeddings: Dict[str, np.ndarray],
    gamma: float,
    alpha: float,
) -> List[Tuple[str, List[float]]]:
    """Score candidate exercises using the latest CogRE scoring function."""
    num_exercises = len(q_matrix)
    all_exercise_embeddings = load_all_exercise_embeddings(q_matrix, entity_embeddings)
    rec_embedding = get_relation_embedding("rec", relation_embeddings).float()
    all_exercise_projection = apply_cogre_tail(all_exercise_embeddings, rec_embedding.unsqueeze(0))

    uid_ex_scores = []
    for uid in uid_mlkc_dict.keys():
        concept_keys = list(uid_mlkc_dict[uid].keys())
        concept_count = max(len(concept_keys), 1)

        if concept_keys:
            concept_embeddings = torch.stack([
                get_entity_embedding(concept, entity_embeddings)
                for concept in concept_keys
            ]).float()
            mlkc_embeddings = torch.stack([
                get_relation_embedding(uid_mlkc_dict[uid].get(concept, "mlkc0.00"), relation_embeddings, "mlkc0.00")
                for concept in concept_keys
            ]).float()
            pkc_embeddings = torch.stack([
                get_relation_embedding(uid_pkc_dict[uid].get(concept, "pkc0.00"), relation_embeddings, "pkc0.00")
                for concept in concept_keys
            ]).float()

            hr_mlkc = apply_cogre_head(concept_embeddings, mlkc_embeddings)
            hr_pkc = apply_cogre_head(concept_embeddings, pkc_embeddings)

            dist_mlkc = calc_alpha_l1(all_exercise_projection.unsqueeze(0) - hr_mlkc.unsqueeze(1), alpha)
            dist_pkc = calc_alpha_l1(all_exercise_projection.unsqueeze(0) - hr_pkc.unsqueeze(1), alpha)
            fr1 = (gamma - dist_mlkc).sum(dim=0) + (gamma - dist_pkc).sum(dim=0)
        else:
            fr1 = torch.zeros(num_exercises)

        exfr_embeddings = torch.stack([
            get_relation_embedding(uid_exfr_dict[uid].get(f"ex{qid}", "exfr0.00"), relation_embeddings, "exfr0.00")
            for qid in range(num_exercises)
        ]).float()
        state_efr = apply_cogre_head(all_exercise_embeddings, exfr_embeddings)
        state_efr_rec = apply_cogre_head(state_efr, rec_embedding.unsqueeze(0))
        dist_efr = calc_alpha_l1(all_exercise_projection - state_efr_rec, alpha)
        fr2 = gamma - dist_efr

        final_scores = (fr1 / concept_count) + fr2
        uid_ex_scores.append((uid, final_scores.tolist()))

    return uid_ex_scores


def score_candidates_triplere(
    uid_mlkc_dict: dict,
    uid_pkc_dict: dict,
    uid_exfr_dict: dict,
    q_matrix: List[List[int]],
    entity_embeddings: Dict[str, np.ndarray],
    relation_embeddings: Dict[str, np.ndarray],
    gamma: float,
    triplere_u: float,
) -> List[Tuple[str, List[float]]]:
    """Score candidate exercises using the TripleRE scoring function."""
    num_exercises = len(q_matrix)
    all_exercise_embeddings = load_all_exercise_embeddings(q_matrix, entity_embeddings)
    rec_embedding = get_relation_embedding("rec", relation_embeddings).float()

    r_h_rec, r_t_rec, r_m_rec = torch.chunk(rec_embedding, 3, dim=-1)
    all_exercise_projection = all_exercise_embeddings * (r_t_rec + triplere_u)

    def project_head(entity_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        r_h, _, r_m = torch.chunk(rel_emb, 3, dim=-1)
        return entity_emb * (r_h + triplere_u) + r_m

    def project_with_rec(state_emb: torch.Tensor) -> torch.Tensor:
        return state_emb * (r_h_rec + triplere_u) + r_m_rec

    uid_ex_scores = []
    for uid in uid_mlkc_dict.keys():
        concept_keys = list(uid_mlkc_dict[uid].keys())
        concept_count = max(len(concept_keys), 1)

        if concept_keys:
            concept_embeddings = torch.stack([
                get_entity_embedding(concept, entity_embeddings)
                for concept in concept_keys
            ]).float()
            mlkc_embeddings = torch.stack([
                get_relation_embedding(uid_mlkc_dict[uid].get(concept, "mlkc0.00"), relation_embeddings, "mlkc0.00")
                for concept in concept_keys
            ]).float()
            pkc_embeddings = torch.stack([
                get_relation_embedding(uid_pkc_dict[uid].get(concept, "pkc0.00"), relation_embeddings, "pkc0.00")
                for concept in concept_keys
            ]).float()

            state_mlkc = project_head(concept_embeddings, mlkc_embeddings)
            state_pkc = project_head(concept_embeddings, pkc_embeddings)
            hr_all = torch.cat([project_with_rec(state_mlkc), project_with_rec(state_pkc)], dim=0)
            dist = torch.cdist(hr_all, all_exercise_projection, p=2.0)
            fr1 = (gamma - dist).sum(dim=0)
        else:
            fr1 = torch.zeros(num_exercises)

        exfr_embeddings = torch.stack([
            get_relation_embedding(uid_exfr_dict[uid].get(f"ex{qid}", "exfr0.00"), relation_embeddings, "exfr0.00")
            for qid in range(num_exercises)
        ]).float()
        state_efr = project_head(all_exercise_embeddings, exfr_embeddings)
        state_efr_rec = project_with_rec(state_efr)
        dist_efr = torch.norm(state_efr_rec - all_exercise_projection, p=2.0, dim=1)
        fr2 = gamma - dist_efr

        final_scores = (fr1 / concept_count) + fr2
        uid_ex_scores.append((uid, final_scores.tolist()))

    return uid_ex_scores


def score_candidates_transe_adv(
    uid_mlkc_dict: dict,
    uid_pkc_dict: dict,
    uid_exfr_dict: dict,
    q_matrix: List[List[int]],
    entity_embeddings: Dict[str, np.ndarray],
    relation_embeddings: Dict[str, np.ndarray],
    gamma: float,
) -> List[Tuple[str, List[float]]]:
    """Score candidate exercises using the original TransE-ADV evaluation rule."""
    num_exercises = len(q_matrix)
    all_exercise_embeddings = load_all_exercise_embeddings(q_matrix, entity_embeddings)
    rec_embedding = get_relation_embedding("rec", relation_embeddings).float()

    uid_ex_scores = []
    for uid in uid_mlkc_dict.keys():
        concept_keys = list(uid_mlkc_dict[uid].keys())
        concept_count = max(len(concept_keys), 1)

        if concept_keys:
            concept_embeddings = torch.stack([
                get_entity_embedding(concept, entity_embeddings)
                for concept in concept_keys
            ]).float()
            mlkc_embeddings = torch.stack([
                get_relation_embedding(uid_mlkc_dict[uid].get(concept, "mlkc0.00"), relation_embeddings, "mlkc0.00")
                for concept in concept_keys
            ]).float()
            pkc_embeddings = torch.stack([
                get_relation_embedding(uid_pkc_dict[uid].get(concept, "pkc0.00"), relation_embeddings, "pkc0.00")
                for concept in concept_keys
            ]).float()

            dist_mlkc = torch.cdist(concept_embeddings + mlkc_embeddings + rec_embedding, all_exercise_embeddings, p=2.0)
            dist_pkc = torch.cdist(concept_embeddings + pkc_embeddings + rec_embedding, all_exercise_embeddings, p=2.0)
            fr1 = (gamma - dist_mlkc).sum(dim=0) + (gamma - dist_pkc).sum(dim=0)
        else:
            fr1 = torch.zeros(num_exercises)

        exfr_embeddings = torch.stack([
            get_relation_embedding(uid_exfr_dict[uid].get(f"ex{qid}", "exfr0.00"), relation_embeddings, "exfr0.00")
            for qid in range(num_exercises)
        ]).float()
        dist_efr = torch.norm(all_exercise_embeddings + exfr_embeddings + rec_embedding - all_exercise_embeddings, p=2.0, dim=1)
        fr2 = gamma - dist_efr

        final_scores = (fr1 / concept_count) + fr2
        uid_ex_scores.append((uid, final_scores.tolist()))

    return uid_ex_scores


def score_candidates(
    model_type: str,
    uid_mlkc_dict: dict,
    uid_pkc_dict: dict,
    uid_exfr_dict: dict,
    q_matrix: List[List[int]],
    entity_embeddings: Dict[str, np.ndarray],
    relation_embeddings: Dict[str, np.ndarray],
    gamma: float,
    cogre_alpha: float,
    triplere_u: float,
) -> List[Tuple[str, List[float]]]:
    """Dispatch candidate scoring to the selected model backend."""
    if model_type == "cogre":
        return score_candidates_cogre(
            uid_mlkc_dict,
            uid_pkc_dict,
            uid_exfr_dict,
            q_matrix,
            entity_embeddings,
            relation_embeddings,
            gamma,
            cogre_alpha,
        )

    if model_type == "triplere":
        return score_candidates_triplere(
            uid_mlkc_dict,
            uid_pkc_dict,
            uid_exfr_dict,
            q_matrix,
            entity_embeddings,
            relation_embeddings,
            gamma,
            triplere_u,
        )

    if model_type == "transe_adv":
        return score_candidates_transe_adv(
            uid_mlkc_dict,
            uid_pkc_dict,
            uid_exfr_dict,
            q_matrix,
            entity_embeddings,
            relation_embeddings,
            gamma,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def load_offline_stats(stats_path: Path) -> Tuple[Dict[int, float], float]:
    """Load offline exercise difficulty statistics."""
    with stats_path.open("rb") as file:
        stats_data = pickle.load(file)

    return stats_data["exercise_df_dict"], stats_data["global_df_mean"]


def load_student_history(csv_path: Path) -> Tuple[Dict[str, Dict[int, float]], float]:
    """Load student concept-level historical correctness from a pyKT CSV.

    This function keeps the same response handling as the old evaluation script:
    every numeric response is used in the history statistics.
    """
    student_kc_acc_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    global_correct = 0
    global_total = 0

    try:
        with csv_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                uid = str(row.get("uid", "")).strip()
                concepts_str = str(row.get("concepts", "")).replace("nan", "")
                responses_str = str(row.get("responses", "")).replace("nan", "")

                if not uid or not concepts_str or not responses_str:
                    continue

                concepts = concepts_str.split(",")
                responses = responses_str.split(",")
                sequence_length = min(len(concepts), len(responses))

                for idx in range(sequence_length):
                    concept_value = concepts[idx].strip()
                    response_value = responses[idx].strip()

                    if not concept_value or not response_value:
                        continue

                    try:
                        concept_id = int(float(concept_value))
                        correct = int(float(response_value))
                    except ValueError:
                        continue

                    student_kc_acc_dict[uid][concept_id].append(correct)
                    global_correct += correct
                    global_total += 1

    except Exception:
        return {}, 0.5

    final_student_kc_acc = {}
    for uid, concept_data in student_kc_acc_dict.items():
        final_student_kc_acc[uid] = {
            concept_id: sum(responses) / len(responses)
            for concept_id, responses in concept_data.items()
        }

    global_history_mean = global_correct / global_total if global_total > 0 else 0.5
    return final_student_kc_acc, global_history_mean


def acc_dynamic_irt(
    uid_mlkc_dict: dict,
    uid_ex_scores: List[Tuple[str, List[float]]],
    q_matrix: List[List[int]],
    n: int,
    student_kc_acc: Dict[str, Dict[int, float]],
    exercise_df_dict: Dict[int, float],
    global_history_mean: float,
    global_df_mean: float,
) -> Tuple[float, float]:
    """Compute ACC@N using an IRT-style objective success probability."""
    acc_values = []

    for uid, scores in uid_ex_scores:
        sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        top_exercise_ids = [exercise_id for exercise_id, _ in sorted_scores[:n]]
        user_mlkc = uid_mlkc_dict.get(uid, {})

        score_sum = 0.0
        for exercise_id in top_exercise_ids:
            concept_ids = [
                idx for idx, value in enumerate(q_matrix[exercise_id])
                if value == 1
            ]
            concept_count = len(concept_ids)

            df_e = exercise_df_dict.get(exercise_id, global_df_mean)
            df_e_safe = np.clip(df_e, 0.05, 0.95)
            difficulty = 1.0 - df_e_safe
            b_e = np.log(difficulty / (1.0 - difficulty))

            if concept_count > 0:
                history = student_kc_acc.get(str(uid), {})
                student_ability = sum(
                    history.get(concept_id, global_history_mean)
                    for concept_id in concept_ids
                ) / concept_count
            else:
                student_ability = global_history_mean

            ability_safe = np.clip(student_ability, 0.05, 0.95)
            theta_ie = np.log(ability_safe / (1.0 - ability_safe))
            objective_success_prob = 1.0 / (1.0 + np.exp(-(theta_ie - b_e)))

            model_mastery_product = 1.0
            for concept_id in concept_ids:
                concept_key = f"kc{concept_id}"
                if concept_key in user_mlkc:
                    prob = parse_probability_token(user_mlkc[concept_key], "mlkc", default=global_history_mean)
                else:
                    prob = global_history_mean
                model_mastery_product *= prob

            if concept_count > 0:
                predicted_mastery = model_mastery_product ** (1.0 / concept_count)
            else:
                predicted_mastery = global_history_mean

            score_sum += 1.0 - np.abs(objective_success_prob - predicted_mastery)

        acc_values.append(score_sum / n)

    return float(np.mean(acc_values)), float(np.std(acc_values))


def emcc_score_zpd(
    uid_pkc_dict: dict,
    uid_ex_scores: List[Tuple[str, List[float]]],
    q_matrix: List[List[int]],
    n: int,
    student_kc_acc: Dict[str, Dict[int, float]],
    exercise_df_dict: Dict[int, float],
    global_history_mean: float,
    global_df_mean: float,
    beta: float = 1.0,
    normalize: bool = True,
) -> Tuple[float, float]:
    """Compute raw or local-normalized ZPD-EMCC@N."""
    emcc_values = []
    gamma_scale = 2.0

    for uid, scores in uid_ex_scores:
        sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        top_exercise_ids = [exercise_id for exercise_id, _ in sorted_scores[:n]]

        user_pkc = uid_pkc_dict.get(uid, {})
        history = student_kc_acc.get(str(uid), {})
        concept_alphas = collections.defaultdict(list)

        for exercise_id in top_exercise_ids:
            concept_ids = [
                idx for idx, value in enumerate(q_matrix[exercise_id])
                if value == 1
            ]
            concept_count = len(concept_ids)
            if concept_count == 0:
                continue

            df_e = exercise_df_dict.get(exercise_id, global_df_mean)
            df_e_safe = np.clip(df_e, 0.05, 0.95)
            normalized_difficulty = 1.0 - df_e_safe

            exercise_ability = sum(
                history.get(concept_id, global_history_mean)
                for concept_id in concept_ids
            ) / concept_count
            exercise_ability_safe = np.clip(exercise_ability, 0.05, 0.95)
            appropriateness = 1.0 - np.abs(exercise_ability_safe - normalized_difficulty)

            for concept_id in concept_ids:
                concept_mastery = history.get(concept_id, global_history_mean)
                concept_mastery_safe = np.clip(concept_mastery, 0.05, 0.95)
                logit = gamma_scale * (appropriateness - concept_mastery_safe)
                alpha_iek = 1.0 / (1.0 + np.exp(-logit))
                concept_alphas[concept_id].append(alpha_iek)

        raw_expected_coverage = 0.0
        covered_need = 0.0

        for concept_id, alphas in concept_alphas.items():
            concept_key = f"kc{concept_id}"
            pkc_value = parse_probability_token(user_pkc.get(concept_key), "pkc", default=0.0)
            f_k_s = 1.0 - np.exp(-beta * np.sum(alphas))

            raw_expected_coverage += pkc_value * f_k_s
            covered_need += pkc_value

        if normalize:
            local_emcc = raw_expected_coverage / covered_need if covered_need > 0 else 0.0
            emcc_values.append(local_emcc)
        else:
            emcc_values.append(raw_expected_coverage)

    return float(np.mean(emcc_values)), float(np.std(emcc_values))


def print_metric(metric_name: str, n: int, mean_value: float, std_value: float) -> None:
    """Print one metric in a compact format."""
    print(f"n={n} | {metric_name} mean={mean_value:.4f}, std={std_value:.4f}")


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    embedding_path = Path(args.embedding_path)
    test_triples_path = resolve_path(args.test_triples_path, data_path / "akt_fpkc_h5_g09_test_triples.txt")
    offline_stats_path = resolve_path(args.offline_stats_path, data_path / "raw_stats_akt.pkl")
    student_history_path = Path(args.student_history_path)

    print("开始测试。")
    print(f"模型类型：{args.model_type}")
    print(f"模型目录：{embedding_path}")

    entity_embedding, relation_embedding = load_embeddings(embedding_path)
    q_matrix = load_q_matrix(data_path)
    entity2id = load_id_dict(data_path / "entities.dict")
    relation2id = load_id_dict(data_path / "relations.dict")
    entity_embeddings, relation_embeddings = build_embedding_dicts(
        entity_embedding,
        relation_embedding,
        entity2id,
        relation2id,
    )
    uid_mlkc_dict, uid_pkc_dict, uid_exfr_dict = load_test_triples(test_triples_path)

    print("正在生成候选习题得分。")
    uid_ex_scores = score_candidates(
        args.model_type,
        uid_mlkc_dict,
        uid_pkc_dict,
        uid_exfr_dict,
        q_matrix,
        entity_embeddings,
        relation_embeddings,
        args.gamma,
        args.cogre_alpha,
        args.triplere_u,
    )

    print("正在加载评估基准。")
    exercise_df_dict, global_df_mean = load_offline_stats(offline_stats_path)
    student_kc_acc, global_history_mean = load_student_history(student_history_path)

    print("开始计算指标。")
    for n in args.top_n:
        mean_acc, std_acc = acc_dynamic_irt(
            uid_mlkc_dict,
            uid_ex_scores,
            q_matrix,
            n,
            student_kc_acc,
            exercise_df_dict,
            global_history_mean,
            global_df_mean,
        )
        print_metric("ACC", n, mean_acc, std_acc)

        local_mean_emcc, local_std_emcc = emcc_score_zpd(
            uid_pkc_dict,
            uid_ex_scores,
            q_matrix,
            n,
            student_kc_acc,
            exercise_df_dict,
            global_history_mean,
            global_df_mean,
            beta=1.0,
            normalize=True,
        )
        print_metric("ZPD-EMCC(local)", n, local_mean_emcc, local_std_emcc)

        raw_mean_emcc, raw_std_emcc = emcc_score_zpd(
            uid_pkc_dict,
            uid_ex_scores,
            q_matrix,
            n,
            student_kc_acc,
            exercise_df_dict,
            global_history_mean,
            global_df_mean,
            beta=1.0,
            normalize=False,
        )
        print_metric("ZPD-EMCC(raw)", n, raw_mean_emcc, raw_std_emcc)

    print("测试完成。")


if __name__ == "__main__":
    main()
