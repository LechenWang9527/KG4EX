import collections
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

csv.field_size_limit(sys.maxsize)

DATA_PATH = Path("../data/algebra2005")
EMBEDDING_PATH = Path("./models/algebra2005/cog_h1_g08")
TEST_TRIPLES_PATH = DATA_PATH / "akt_fpkc_h1_g08_test_triples.txt"
OFFLINE_STATS_PATH = DATA_PATH / "raw_stats_akt.pkl"
STUDENT_HISTORY_PATH = Path("../pyKT_example/output_final_akt.csv")
UID_KC_RESPONSE_PATH = DATA_PATH / "algebra2005_uid_kc_response.txt"

GAMMA = 12.0
COGRE_ALPHA = 0.4
TOP_N_VALUES = (10, 20)


def load_embeddings(embedding_path):
    try:
        relation_embedding = np.load(embedding_path / "relation_embedding.npy")
        entity_embedding = np.load(embedding_path / "entity_embedding.npy")
    except Exception as exc:
        raise FileNotFoundError(f"Failed to load embeddings from {embedding_path}") from exc

    return entity_embedding, relation_embedding


def load_q_matrix(data_path):
    q_matrix = []
    with (data_path / "Q.txt").open("r", encoding="utf-8") as file:
        for line in file:
            concepts = line.strip().split(",")
            q_matrix.append([int(value) for value in concepts])
    return q_matrix


def load_id_dict(path):
    id_dict = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            idx, key = line.strip().split("\t")
            id_dict[key] = int(idx)
    return id_dict


def build_embedding_dicts(entity_embedding, relation_embedding, entity2id, relation2id):
    entity_embeddings = {
        entity: entity_embedding[idx, :]
        for entity, idx in entity2id.items()
    }
    relation_embeddings = {
        relation: relation_embedding[idx, :]
        for relation, idx in relation2id.items()
    }
    return entity_embeddings, relation_embeddings


def load_test_triples(test_path):
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test triples file: {test_path}")

    uid_mlkc_dict = {}
    uid_pkc_dict = {}
    uid_exfr_dict = {}

    with test_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue

            item, relation, uid = parts
            uid_mlkc_dict.setdefault(uid, {})
            uid_pkc_dict.setdefault(uid, {})
            uid_exfr_dict.setdefault(uid, {})

            if relation.startswith("mlkc"):
                uid_mlkc_dict[uid][item] = relation
            elif relation.startswith("exfr"):
                uid_exfr_dict[uid][item] = relation
            elif relation.startswith("pkc"):
                uid_pkc_dict[uid][item] = relation

    return uid_mlkc_dict, uid_pkc_dict, uid_exfr_dict


def get_relation_embedding(key, relation_embeddings, default="rec"):
    if key.endswith(".0") and not key.endswith(".00"):
        key += "0"

    if key in relation_embeddings:
        return torch.from_numpy(relation_embeddings[key])

    if default in relation_embeddings:
        return torch.from_numpy(relation_embeddings[default])

    first_embedding = next(iter(relation_embeddings.values()))
    return torch.zeros_like(torch.from_numpy(first_embedding))


def get_entity_embedding(key, entity_embeddings):
    if key in entity_embeddings:
        return torch.from_numpy(entity_embeddings[key])

    first_embedding = next(iter(entity_embeddings.values()))
    return torch.zeros_like(torch.from_numpy(first_embedding))


def apply_cogre_head(head_emb, rel_emb):
    """
    Relation-gated head projection:
    h_proj = h * (sigmoid(r_gate_h) + gamma_r) + r_m
    """
    r_gate_h, _, r_m, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_h = torch.sigmoid(r_gate_h)
    gamma_r = 0.2 * torch.sigmoid(r_res)
    return head_emb * (gate_h + gamma_r) + r_m


def apply_cogre_tail(tail_emb, rel_emb):
    """
    Relation-gated tail projection:
    t_proj = t * (sigmoid(r_gate_t) + gamma_r)
    """
    _, r_gate_t, _, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_t = torch.sigmoid(r_gate_t)
    gamma_r = 0.2 * torch.sigmoid(r_res)
    return tail_emb * (gate_t + gamma_r)


def calc_alpha_l1(diff_tensor, alpha=COGRE_ALPHA):
    """
    Alpha-weighted asymmetric L1 penalty.

    diff > 0  : full penalty
    diff <= 0 : alpha-scaled penalty
    """
    abs_diff = torch.abs(diff_tensor)
    penalty = torch.where(diff_tensor > 0, abs_diff, alpha * abs_diff)
    return torch.sum(penalty, dim=-1)


def score_candidates(uid_mlkc_dict, uid_pkc_dict, uid_exfr_dict, q_matrix,
                     entity_embeddings, relation_embeddings):
    num_exercises = len(q_matrix)

    all_exercise_embeddings = torch.stack([
        get_entity_embedding(f"ex{qid}", entity_embeddings)
        for qid in range(num_exercises)
    ]).float()
    rec_embedding = get_relation_embedding("rec", relation_embeddings).float()
    all_exercise_projection = apply_cogre_tail(
        all_exercise_embeddings,
        rec_embedding.unsqueeze(0),
    )

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
                get_relation_embedding(
                    uid_mlkc_dict[uid].get(concept, "mlkc0.00"),
                    relation_embeddings,
                    "mlkc0.00",
                )
                for concept in concept_keys
            ]).float()

            pkc_embeddings = torch.stack([
                get_relation_embedding(
                    uid_pkc_dict[uid].get(concept, "pkc0.00"),
                    relation_embeddings,
                    "pkc0.00",
                )
                for concept in concept_keys
            ]).float()

            hr_mlkc = apply_cogre_head(concept_embeddings, mlkc_embeddings)
            hr_pkc = apply_cogre_head(concept_embeddings, pkc_embeddings)

            dist_mlkc = calc_alpha_l1(
                hr_mlkc.unsqueeze(1) - all_exercise_projection.unsqueeze(0)
            )
            dist_pkc = calc_alpha_l1(
                hr_pkc.unsqueeze(1) - all_exercise_projection.unsqueeze(0)
            )

            fr1 = (GAMMA - dist_mlkc).sum(dim=0) + (GAMMA - dist_pkc).sum(dim=0)
        else:
            fr1 = torch.zeros(num_exercises)

        exfr_embeddings = torch.stack([
            get_relation_embedding(
                uid_exfr_dict[uid].get(f"ex{qid}", "exfr0.00"),
                relation_embeddings,
                "exfr0.00",
            )
            for qid in range(num_exercises)
        ]).float()

        state_efr = apply_cogre_head(all_exercise_embeddings, exfr_embeddings)
        state_efr_rec = apply_cogre_head(state_efr, rec_embedding.unsqueeze(0))

        dist_efr = calc_alpha_l1(state_efr_rec - all_exercise_projection)
        fr2 = GAMMA - dist_efr

        final_scores = (fr1 / concept_count) + fr2
        uid_ex_scores.append((uid, final_scores.tolist()))

    return uid_ex_scores


def load_offline_stats(stats_path):
    with stats_path.open("rb") as file:
        stats_data = pickle.load(file)

    return stats_data["exercise_df_dict"], stats_data["global_df_mean"]


def load_student_history(csv_path):
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
                        kc_id = int(float(concept_value))
                        correct = int(float(response_value))
                    except ValueError:
                        continue

                    student_kc_acc_dict[uid][kc_id].append(correct)
                    global_correct += correct
                    global_total += 1
    except Exception:
        return {}, 0.5

    final_student_kc_acc = {}
    for uid, kc_data in student_kc_acc_dict.items():
        final_student_kc_acc[uid] = {
            kc_id: sum(responses) / len(responses)
            for kc_id, responses in kc_data.items()
        }

    global_history_mean = global_correct / global_total if global_total > 0 else 0.5
    return final_student_kc_acc, global_history_mean


def acc_dynamic_irt(uid_mlkc_dict, uid_ex_scores, q_matrix, n, student_kc_acc,
                    exercise_df_dict, global_history_mean, global_df_mean):
    acc_values = []

    for uid, scores in uid_ex_scores:
        sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        top_exercise_ids = [exercise_id for exercise_id, _ in sorted_scores[:n]]
        user_mlkc = uid_mlkc_dict.get(uid, {})

        diff_sum = 0.0
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
                local_history_sum = sum(
                    history.get(concept_id, global_history_mean)
                    for concept_id in concept_ids
                )
                student_ability = local_history_sum / concept_count
            else:
                student_ability = global_history_mean

            ability_safe = np.clip(student_ability, 0.05, 0.95)
            theta_ie = np.log(ability_safe / (1.0 - ability_safe))
            objective_success_prob = 1.0 / (1.0 + np.exp(-(theta_ie - b_e)))

            model_mastery_product = 1.0
            for concept_id in concept_ids:
                concept_key = f"kc{concept_id}"
                if concept_key in user_mlkc:
                    try:
                        prob = float(user_mlkc[concept_key].replace("mlkc", ""))
                    except ValueError:
                        prob = global_history_mean
                else:
                    prob = global_history_mean

                model_mastery_product *= prob

            if concept_count > 0:
                predicted_mastery = model_mastery_product ** (1.0 / concept_count)
            else:
                predicted_mastery = global_history_mean

            diff_sum += 1.0 - np.abs(objective_success_prob - predicted_mastery)

        acc_values.append(diff_sum / n)

    return np.mean(acc_values), np.std(acc_values)


def novelty(uid_kc_response, uid_ex_scores, q_matrix, n):
    novelty_values = []

    for uid, scores in uid_ex_scores:
        sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        top_exercise_ids = [exercise_id for exercise_id, _ in sorted_scores[:n]]
        historical_concepts = set(uid_kc_response.get(uid, []))

        novelty_sum = 0.0
        for exercise_id in top_exercise_ids:
            exercise_concepts = {
                idx for idx, value in enumerate(q_matrix[exercise_id])
                if value == 1
            }
            union = historical_concepts.union(exercise_concepts)
            if union:
                intersection = historical_concepts.intersection(exercise_concepts)
                novelty_sum += 1.0 - len(intersection) / len(union)

        novelty_values.append(novelty_sum / n)

    return np.mean(novelty_values), np.std(novelty_values)


def emcc_score_zpd(uid_pkc_dict, uid_ex_scores, q_matrix, n, student_kc_acc,
                   exercise_df_dict, global_history_mean, global_df_mean, beta=1.0):
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

            local_history_sum = sum(
                history.get(concept_id, global_history_mean)
                for concept_id in concept_ids
            )
            exercise_ability = local_history_sum / concept_count
            exercise_ability_safe = np.clip(exercise_ability, 0.05, 0.95)

            appropriateness = 1.0 - np.abs(exercise_ability_safe - normalized_difficulty)

            for concept_id in concept_ids:
                concept_mastery = history.get(concept_id, global_history_mean)
                concept_mastery_safe = np.clip(concept_mastery, 0.05, 0.95)
                logit = gamma_scale * (appropriateness - concept_mastery_safe)
                alpha_iek = 1.0 / (1.0 + np.exp(-logit))
                concept_alphas[concept_id].append(alpha_iek)

        expected_coverage = 0.0
        for concept_id, alphas in concept_alphas.items():
            concept_key = f"kc{concept_id}"
            pkc_value = 0.0

            if concept_key in user_pkc:
                try:
                    pkc_value = float(user_pkc[concept_key].replace("pkc", ""))
                except ValueError:
                    pkc_value = 0.0

            f_k_s = 1.0 - np.exp(-beta * np.sum(alphas))
            expected_coverage += pkc_value * f_k_s

        emcc_values.append(expected_coverage)

    return np.mean(emcc_values), np.std(emcc_values)


# Backward-compatible aliases for earlier scripts.
ACC_Dynamic_IRT = acc_dynamic_irt
Nov = novelty
EMCC_Score_ZPD = emcc_score_zpd


def load_uid_kc_response(path):
    uid_kc_response = {}
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                uid, response_text = line.strip().split("\t")
                uid_kc_response[uid] = [int(value) for value in response_text.split(",")]
    except Exception as exc:
        print(f"Failed to load uid-kc response file: {exc}")

    return uid_kc_response


def print_metric(metric_name, n, mean_value, std_value):
    print(f"n={n} | {metric_name} mean={mean_value:.4f}, std={std_value:.4f}")


def main():
    print("Loading data...")
    entity_embedding, relation_embedding = load_embeddings(EMBEDDING_PATH)
    q_matrix = load_q_matrix(DATA_PATH)
    entity2id = load_id_dict(DATA_PATH / "entities.dict")
    relation2id = load_id_dict(DATA_PATH / "relations.dict")
    entity_embeddings, relation_embeddings = build_embedding_dicts(
        entity_embedding,
        relation_embedding,
        entity2id,
        relation2id,
    )
    uid_mlkc_dict, uid_pkc_dict, uid_exfr_dict = load_test_triples(TEST_TRIPLES_PATH)

    print("Scoring candidates...")
    uid_ex_scores = score_candidates(
        uid_mlkc_dict,
        uid_pkc_dict,
        uid_exfr_dict,
        q_matrix,
        entity_embeddings,
        relation_embeddings,
    )
    print("Candidate scoring complete.")

    print("Loading evaluation baselines...")
    exercise_df_dict, global_df_mean = load_offline_stats(OFFLINE_STATS_PATH)
    student_kc_acc, global_history_mean = load_student_history(STUDENT_HISTORY_PATH)

    print("Calculating ACC...")
    for n in TOP_N_VALUES:
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

    print("Calculating NOV...")
    all_uid_kc_response = load_uid_kc_response(UID_KC_RESPONSE_PATH)
    test_uid_kc_response = {
        uid: all_uid_kc_response.get(uid, [])
        for uid in uid_mlkc_dict.keys()
    }
    for n in TOP_N_VALUES:
        mean_nov, std_nov = novelty(test_uid_kc_response, uid_ex_scores, q_matrix, n)
        print_metric("NOV", n, mean_nov, std_nov)

    print("Calculating ZPD-EMCC...")
    for n in TOP_N_VALUES:
        mean_emcc, std_emcc = emcc_score_zpd(
            uid_pkc_dict,
            uid_ex_scores,
            q_matrix,
            n,
            student_kc_acc,
            exercise_df_dict,
            global_history_mean,
            global_df_mean,
            beta=1.0,
        )
        print_metric("ZPD-EMCC", n, mean_emcc, std_emcc)


if __name__ == "__main__":
    main()
