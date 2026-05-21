<<<<<<< HEAD
import os
import shutil

# 1. 配置你的输入输出路径
INPUT_TRAIN = "dkt_fpkc_h5_g08_triples.txt"      # 训练集
INPUT_TEST = "dkt_fpkc_h5_g08_test_triples.txt"      # 🚨 注意：这里改成了你测试代码里用的名字，请确保文件名一致！
OUTPUT_DIR = "."   # 统一输出到数据文件夹

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("🚀 开始构建【完备强同步字典】...")
print("="*60)

entities = set()
relations = set()

# ==========================================
# 2. 扫描并收集训练集
# ==========================================
if os.path.exists(INPUT_TRAIN):
    print(f"✅ [1/2] 正在扫描训练集: {INPUT_TRAIN}")
    with open(INPUT_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # 🔪 致命修复：强制剥离每一个元素可能携带的隐藏空格
                h, r, t = [p.strip() for p in parts]
                entities.add(h)
                entities.add(t)
                relations.add(r)
                
    # 🌟 致命修复：强行把这个训练集拷贝到 OUTPUT_DIR，确保 run.py 读到的绝对是和字典配套的文本！
    try:
        shutil.copy(INPUT_TRAIN, os.path.join(OUTPUT_DIR, "dkt_fpkc_h5_g08_triples.txt"))
        print(f"   -> 已将 {INPUT_TRAIN} 同步至输出目录！")
    except shutil.SameFileError:
        print(f"   -> {INPUT_TRAIN} 已在当前目录，无需移动！")
else:
    print(f"❌ 找不到训练集: {INPUT_TRAIN}")

# ==========================================
# 3. 扫描并收集测试集
# ==========================================
if os.path.exists(INPUT_TEST):
    print(f"✅ [2/2] 正在扫描测试集: {INPUT_TEST}")
    with open(INPUT_TEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) == 3:
                # 测试集格式 item1, item2(关系), uid
                item1, item2, uid = [p.strip() for p in parts]
                entities.add(item1)
                entities.add(uid)
                relations.add(item2)
                
    # 🌟 同样把测试集也拷贝过去，保持绝对同步
    # 🌟 修复：防呆设计
    try:
        shutil.copy(INPUT_TEST, os.path.join(OUTPUT_DIR, "dkt_fpkc_h5_g08_test_triples.txt"))
        print(f"   -> 已将 {INPUT_TEST} 同步至输出目录！")
    except shutil.SameFileError:
        print(f"   -> {INPUT_TEST} 已在当前目录，无需移动！")
else:
    print(f"⚠️ 找不到测试集: {INPUT_TEST} (请检查文件名是否对应)")

# ==========================================
# 4. 生成规范字典
# ==========================================
entities = sorted(list(entities))
relations = sorted(list(relations))

entity2id = {ent: i for i, ent in enumerate(entities)}
relation2id = {rel: i for i, rel in enumerate(relations)}

print("-" * 60)
print(f"📊 统计结果：")
print(f"   -> 独立实体总数 : {len(entities)}")
print(f"   -> 独立关系总数 : {len(relations)}")

# 生成 entities.dict
with open(os.path.join(OUTPUT_DIR, "entities.dict"), "w", encoding="utf-8") as f:
    for ent, eid in entity2id.items():
        f.write(f"{eid}\t{ent}\n")

# 生成 relations.dict
with open(os.path.join(OUTPUT_DIR, "relations.dict"), "w", encoding="utf-8") as f:
    for rel, rid in relation2id.items():
        f.write(f"{rid}\t{rel}\n")

print("-" * 60)
print(f"🎉 完美收工！数据和字典已在 {OUTPUT_DIR}/ 实现 100% 强同步。")
print("👉 现在去跑 python run.py，绝对不会再报 KeyError 了！")
=======
#!/usr/bin/env python3
"""
Generate KG4EX entity and relation dictionaries from triple files.

The script preserves raw entity and relation labels from the input triples.
It does not canonicalize numeric relations or add extra entities from Q.txt.
"""
"""
sh:
python generate_dict.py \
  --train_file akt_fpkc_h5_g09_triples.txt \
  --test_file akt_fpkc_h5_g09_test_triples.txt \
  --q_file Q.txt
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate entities.dict and relations.dict for KG4EX."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="akt_fpkc_h5_g09_triples.txt",
        help="Training triple file name or path.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="akt_fpkc_h5_g09_test_triples.txt",
        help="Test triple file name or path.",
    )
    parser.add_argument(
        "--q_file",
        type=str,
        default="Q.txt",
        help="Reserved for command compatibility. It is not used for dictionary generation.",
    )
    parser.add_argument(
        "--entities_output",
        type=str,
        default="entities.dict",
        help="Output path for entity dictionary.",
    )
    parser.add_argument(
        "--relations_output",
        type=str,
        default="relations.dict",
        help="Output path for relation dictionary.",
    )
    parser.add_argument(
        "--report_output",
        type=str,
        default="generate_dict_report.txt",
        help="Output path for the generation report.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    """Resolve a path relative to the script directory."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def read_triples(triple_path: Path, entities: Set[str], relations: Set[str]) -> Tuple[int, List[str]]:
    """Read triples and collect raw entity and relation labels."""
    if not triple_path.exists():
        raise FileNotFoundError(f"Triple file does not exist: {triple_path}")

    valid_count = 0
    errors = []

    with triple_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            raw_line = line.rstrip("\n")
            stripped = raw_line.strip()

            if not stripped:
                errors.append(f"{triple_path.name}:{line_no}: empty line")
                continue

            parts = stripped.split("\t")
            if len(parts) != 3:
                errors.append(
                    f"{triple_path.name}:{line_no}: expected 3 tab-separated fields, got {len(parts)} | {raw_line}"
                )
                continue

            head, relation, tail = [part.strip() for part in parts]
            if not head or not relation or not tail:
                errors.append(f"{triple_path.name}:{line_no}: empty field detected | {raw_line}")
                continue

            entities.add(head)
            entities.add(tail)
            relations.add(relation)
            valid_count += 1

    return valid_count, errors


def write_dict(items: Iterable[str], output_path: Path) -> int:
    """Write a deterministic id dictionary using lexical order."""
    sorted_items = sorted(set(items))

    with output_path.open("w", encoding="utf-8") as file:
        for idx, item in enumerate(sorted_items):
            file.write(f"{idx}\t{item}\n")

    return len(sorted_items)


def write_report(
    report_path: Path,
    train_path: Path,
    test_path: Path,
    entity_count: int,
    relation_count: int,
    train_valid_count: int,
    test_valid_count: int,
    train_errors: List[str],
    test_errors: List[str],
) -> None:
    """Write a compact dictionary generation report."""
    all_errors = train_errors + test_errors

    with report_path.open("w", encoding="utf-8") as file:
        file.write("KG4EX dictionary generation report\n\n")
        file.write("Input files\n")
        file.write(f"train_file: {train_path}\n")
        file.write(f"test_file: {test_path}\n\n")
        file.write("Summary\n")
        file.write(f"valid_train_triples: {train_valid_count}\n")
        file.write(f"valid_test_triples: {test_valid_count}\n")
        file.write(f"entity_count: {entity_count}\n")
        file.write(f"relation_count: {relation_count}\n")
        file.write(f"malformed_or_invalid_line_count: {len(all_errors)}\n")

        if all_errors:
            file.write("\nMalformed or invalid lines\n")
            for error in all_errors:
                file.write(error + "\n")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    train_path = resolve_path(args.train_file, script_dir)
    test_path = resolve_path(args.test_file, script_dir)
    entities_output = resolve_path(args.entities_output, script_dir)
    relations_output = resolve_path(args.relations_output, script_dir)
    report_output = resolve_path(args.report_output, script_dir)

    print("生成实体字典和关系字典。")
    print(f"训练三元组文件：{train_path}")
    print(f"测试三元组文件：{test_path}")

    entities: Set[str] = set()
    relations: Set[str] = set()

    train_valid_count, train_errors = read_triples(train_path, entities, relations)
    test_valid_count, test_errors = read_triples(test_path, entities, relations)

    entity_count = write_dict(entities, entities_output)
    relation_count = write_dict(relations, relations_output)

    write_report(
        report_output,
        train_path,
        test_path,
        entity_count,
        relation_count,
        train_valid_count,
        test_valid_count,
        train_errors,
        test_errors,
    )

    malformed_count = len(train_errors) + len(test_errors)

    print("字典生成完成。")
    print(f"有效训练三元组数量：{train_valid_count}")
    print(f"有效测试三元组数量：{test_valid_count}")
    print(f"实体数量：{entity_count}")
    print(f"关系数量：{relation_count}")
    print(f"异常行数量：{malformed_count}")
    print(f"实体字典已保存到：{entities_output}")
    print(f"关系字典已保存到：{relations_output}")
    print(f"生成报告已保存到：{report_output}")


if __name__ == "__main__":
    main()
>>>>>>> 36b6bc4 (release: update KG4EX v1.1.0)
