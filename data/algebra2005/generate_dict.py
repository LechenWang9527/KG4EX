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