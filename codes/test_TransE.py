import numpy as np
import torch
import os
import collections
import math
import pickle
import csv
import sys

csv.field_size_limit(sys.maxsize)

dict_path = f"../data/algebra2005"
embedding_path = f"./models/algebra2005/cog_h1_g08"

try:
    relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")
    entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")
except Exception as e:
    exit()

### 读取 Q 矩阵和字典
Q = []
with open(f"{dict_path}/Q.txt", 'r') as file:
    for line in file:
        kc = line.strip().split(',')
        Q.append([int(x) for x in kc])

with open(f"{dict_path}/entities.dict", 'r') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict", 'r') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)

dict_entity_embedding = {ent: entity_embedding[eid, :] for ent, eid in entity2id.items()}
dict_relation_embedding = {rel: relation_embedding[rid, :] for rel, rid in relation2id.items()}



uid_mlkc_dict = {}
uid_pkc_dict = {}
uid_exfr_dict = {}

test_file_path = f"{dict_path}/akt_fpkc_h1_g08_test_triples.txt"
if not os.path.exists(test_file_path):
    exit()

with open(test_file_path, 'r', encoding="UTF-8") as load_file:
    for line in load_file:
        parts = line.strip().split('\t')
        if len(parts) != 3: continue
        
        # 作者原始测试集格式: item(kc/ex) \t relation(mlkc/exfr) \t user(uid)
        item1, item2, uid = parts  
        
        if uid not in uid_mlkc_dict: uid_mlkc_dict[uid] = {}
        if uid not in uid_exfr_dict: uid_exfr_dict[uid] = {}
        if uid not in uid_pkc_dict: uid_pkc_dict[uid] = {}

        if item2.startswith('mlkc'):
            uid_mlkc_dict[uid][item1] = item2
        elif item2.startswith('exfr'):
            uid_exfr_dict[uid][item1] = item2
        elif item2.startswith('pkc'):
            uid_pkc_dict[uid][item1] = item2



def get_rel_emb(key, default='rec'):
    if key.endswith('.0') and not key.endswith('.00'): key += '0'
    if key in dict_relation_embedding: return torch.from_numpy(dict_relation_embedding[key])
    if default in dict_relation_embedding: return torch.from_numpy(dict_relation_embedding[default])
    return torch.zeros_like(torch.from_numpy(list(dict_relation_embedding.values())[0]))

def get_ent_emb(key):
    if key in dict_entity_embedding: return torch.from_numpy(dict_entity_embedding[key])
    return torch.zeros_like(torch.from_numpy(list(dict_entity_embedding.values())[0]))

import torch.nn.functional as F


# ================= 5. CogRE 专属打分逻辑：Alpha 非对称 L1 版 =================
print("⏳ 正在运行 CogRE 模型计算推荐基准得分 (Alpha 非对称 L1 版)...")

uid_ex_scores = []
uid_mlkc_dict_keys_list = list(uid_mlkc_dict.keys())

num_ex = len(Q)
gamma = 12.0

# 必须和 model.py 保持一致
# 当前版本只保留 alpha，不使用 beta
cogre_alpha = 0.4

all_ex_embs = torch.stack([
    get_ent_emb('ex' + str(qid)) for qid in range(num_ex)
]).float()

rec_embedding = get_rel_emb('rec').float()


def apply_cogre_head(head_emb, rel_emb):
    """
    头实体关系门控投影：
    h_proj = h * (sigmoid(r_gate_h) + gamma_r) + r_m
    """
    r_gate_h, _, r_m, r_res = torch.chunk(rel_emb, 4, dim=-1)

    gate_h = torch.sigmoid(r_gate_h)

    # 注意：这里必须和 model.py 保持一致
    # 如果 model.py 里用的是零中心残差，就用这一行
    gamma_r = 0.2 * (torch.sigmoid(r_res))

    # 如果 model.py 里用的是原始正残差，则改成下面这一行：
    # gamma_r = 0.2 * torch.sigmoid(r_res)

    return head_emb * (gate_h + gamma_r) + r_m


def apply_cogre_tail(tail_emb, rel_emb):
    """
    尾实体关系门控投影：
    t_proj = t * (sigmoid(r_gate_t) + gamma_r)
    """
    _, r_gate_t, _, r_res = torch.chunk(rel_emb, 4, dim=-1)

    gate_t = torch.sigmoid(r_gate_t)

    # 注意：这里必须和 model.py 保持一致
    # 如果 model.py 里用的是零中心残差，就用这一行
    gamma_r = 0.2 * (torch.sigmoid(r_res))

    # 如果 model.py 里用的是原始正残差，则改成下面这一行：
    # gamma_r = 0.2 * torch.sigmoid(r_res)

    return tail_emb * (gate_t + gamma_r)


def calc_alpha_l1(diff_tensor):
    """
    Alpha 非对称 L1 惩罚。

    diff > 0  : 全额惩罚
    diff <= 0 : alpha 倍惩罚

    alpha 越小，模型对可接受方向的偏差越宽容，通常有利于 EMCC；
    alpha 越大，模型越接近普通 L1，通常 ACC 更稳。
    """
    abs_d = torch.abs(diff_tensor)

    penalty = torch.where(
        diff_tensor > 0,
        abs_d,
        cogre_alpha * abs_d
    )

    return torch.sum(penalty, dim=-1)


# 所有候选习题经过 rec 关系后的尾实体投影
all_ex_proj = apply_cogre_tail(all_ex_embs, rec_embedding.unsqueeze(0))


for uid in uid_mlkc_dict_keys_list:
    uid_mlkc_keys = list(uid_mlkc_dict[uid].keys())
    mlkc_len = max(len(uid_mlkc_keys), 1)

    if len(uid_mlkc_keys) > 0:
        kc_embs = torch.stack([
            get_ent_emb(kc) for kc in uid_mlkc_keys
        ]).float()

        mlkc_embs = torch.stack([
            get_rel_emb(
                uid_mlkc_dict[uid].get(kc, 'mlkc0.00'),
                'mlkc0.00'
            )
            for kc in uid_mlkc_keys
        ]).float()

        pkc_embs = torch.stack([
            get_rel_emb(
                uid_pkc_dict[uid].get(kc, 'pkc0.00'),
                'pkc0.00'
            )
            for kc in uid_mlkc_keys
        ]).float()

        # 知识点经过 mlkc / pkc 关系后的头实体投影
        # 1. 知识点经过 mlkc / pkc 关系，得到学生当前认知状态
        hr_mlkc = apply_cogre_head(kc_embs, mlkc_embs)
        hr_pkc = apply_cogre_head(kc_embs, pkc_embs)
        
        diff_mlkc = hr_mlkc.unsqueeze(1) - all_ex_proj.unsqueeze(0)
        dist_mlkc = calc_alpha_l1(diff_mlkc)
        
        diff_pkc = hr_pkc.unsqueeze(1) - all_ex_proj.unsqueeze(0)
        dist_pkc = calc_alpha_l1(diff_pkc)

        mlkc_score = (gamma - dist_mlkc).sum(dim=0)
        pkc_score = (gamma - dist_pkc).sum(dim=0)

        fr1 = mlkc_score + pkc_score

    else:
        fr1 = torch.zeros(num_ex)

    # ---------------------------------------------------------
    # 3. 遗忘曲线部分 fr2
    # ---------------------------------------------------------
    efr_embs = torch.stack([
        get_rel_emb(
            uid_exfr_dict[uid].get('ex' + str(qid), 'exfr0.00'),
            'exfr0.00'
        )
        for qid in range(num_ex)
    ]).float()

    # 历史习题经过 exfr 关系投影
    state_efr = apply_cogre_head(all_ex_embs, efr_embs)

    # 再经过 rec 关系投影
    s_efr_rec = apply_cogre_head(state_efr, rec_embedding.unsqueeze(0))

    # 对应公式：
    # diff = h_proj + r_m - t_proj
    diff_efr = s_efr_rec - all_ex_proj
    dist_efr = calc_alpha_l1(diff_efr)

    fr2 = gamma - dist_efr

    # ---------------------------------------------------------
    # 4. 最终推荐分数，大 O 函数保持不变
    # ---------------------------------------------------------
    O_sel = (fr1 / mlkc_len) + fr2

    uid_ex_scores.append((uid, O_sel.tolist()))

print("✅ CogRE 候选得分计算完成。")



'''
# ================= 5.1. CogRE 专属打分逻辑 (纯粹版: 门控 + 自适应残差) =================
uid_ex_scores = []
uid_mlkc_dict_keys_list = list(uid_mlkc_dict.keys())

print(f"一共读取到 {len(uid_mlkc_dict_keys_list)} 个测试学生。")

gamma = 12.0
num_ex = len(Q)

# 提取全体习题的 Entity Embedding [num_ex, 500]
all_ex_embs = torch.stack([get_ent_emb('ex' + str(qid)) for qid in range(num_ex)]).float()
# 提取 Recommendation 关系 Embedding [2000]
rec_embedding = get_rel_emb('rec').float()

# --- 核心算子 1：CogRE 头实体投影 ---
def apply_cogre_head(head_emb, rel_emb):
    """根据纯粹版 CogRE 公式，使用关系门控过滤并转换头实体"""
    r_gate_h, _, r_m, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_h = torch.sigmoid(r_gate_h)
    gamma_r = F.softplus(r_res)
    return head_emb * (gate_h + gamma_r) + r_m

# --- 核心算子 2：CogRE 尾实体投影 ---
def apply_cogre_tail(tail_emb, rel_emb):
    """根据纯粹版 CogRE 公式，使用关系门控过滤并转换尾实体"""
    _, r_gate_t, _, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_t = torch.sigmoid(r_gate_t)
    gamma_r = F.softplus(r_res)
    return tail_emb * (gate_t + gamma_r)

# 预先计算所有候选习题被 rec(推荐) 关系投影后的状态 [num_ex, 500]
all_ex_proj = apply_cogre_tail(all_ex_embs, rec_embedding.unsqueeze(0))

for uid in uid_mlkc_dict_keys_list:
    uid_mlkc_keys = list(uid_mlkc_dict[uid].keys())
    mlkc_len = len(uid_mlkc_keys) if len(uid_mlkc_keys) > 0 else 1
    
    # ---- 步骤 1：计算 fr1 (基于掌握度与预测概率) ----
    if len(uid_mlkc_keys) > 0:
        kc_embs = torch.stack([get_ent_emb(kc) for kc in uid_mlkc_keys]).float()
        mlkc_embs = torch.stack([get_rel_emb(uid_mlkc_dict[uid].get(kc, 'mlkc0.00'), 'mlkc0.00') for kc in uid_mlkc_keys]).float()
        pkc_embs = torch.stack([get_rel_emb(uid_pkc_dict[uid].get(kc, 'pkc0.00'), 'pkc0.00') for kc in uid_mlkc_keys]).float()

        # 1. 知识点经过 mlkc 和 pkc 门控过滤，形成【学生当期状态】
        state_mlkc = apply_cogre_head(kc_embs, mlkc_embs)
        state_pkc = apply_cogre_head(kc_embs, pkc_embs)
        
        # 2. 学生状态叠加 rec 关系，形成【目标习题预期状态】
        hr_mlkc = apply_cogre_head(state_mlkc, rec_embedding.unsqueeze(0))
        hr_pkc = apply_cogre_head(state_pkc, rec_embedding.unsqueeze(0))
        
        # 3. 计算与所有候选习题的欧氏距离
        dist_mlkc = torch.cdist(hr_mlkc, all_ex_proj, p=1.0)
        dist_pkc = torch.cdist(hr_pkc, all_ex_proj, p=1.0)
        
        transE_mlkc = gamma - dist_mlkc
        transE_pkc = gamma - dist_pkc
        
        fr1 = transE_mlkc.sum(dim=0) + transE_pkc.sum(dim=0)
    else:
        fr1 = torch.zeros(num_ex)

    # ---- 步骤 2：计算 fr2 (基于遗忘曲线) ----
    efr_embs = torch.stack([get_rel_emb(uid_exfr_dict[uid].get('ex'+str(qid), 'exfr0.00'), 'exfr0.00') for qid in range(num_ex)]).float()
    
    # 1. 历史习题经过遗忘关系过滤
    state_efr = apply_cogre_head(all_ex_embs, efr_embs)
    # 2. 叠加 rec 关系
    s_efr_rec = apply_cogre_head(state_efr, rec_embedding.unsqueeze(0))
    # 3. 计算自身偏移距离
    dist_efr = torch.norm(s_efr_rec - all_ex_proj, p=1.0, dim=1) 
    fr2 = gamma - dist_efr

    # ---- 步骤 3：合并 O_sel 得分 ----
    O_sel = (fr1 / mlkc_len) + fr2 
    scores = O_sel.tolist() 
    
    uid_ex_scores.append((uid, scores))
'''


'''
# ================= 5.2 生成所有学生的候选打分 (非对称距离版) =================
print("⏳ 正在运行 CogRE 模型计算推荐基准得分 (同步 Leaky ReLU 逻辑)...")
uid_ex_scores = []
num_ex = len(Q)
all_ex_embs = torch.stack([get_ent_emb('ex' + str(qid)) for qid in range(num_ex)]).float()
rec_embedding = get_rel_emb('rec').float()


def apply_cogre_head(head_emb, rel_emb):
    """根据纯粹版 CogRE 公式，使用关系门控过滤并转换头实体"""
    # 将关系向量切分为 4 份
    r_gate_h, _, r_m, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_h = torch.sigmoid(r_gate_h)
    gamma_r = F.softplus(r_res)
    return head_emb * (gate_h + gamma_r) + r_m

def apply_cogre_tail(tail_emb, rel_emb):
    """根据纯粹版 CogRE 公式，使用关系门控过滤并转换尾实体"""
    # 将关系向量切分为 4 份
    _, r_gate_t, _, r_res = torch.chunk(rel_emb, 4, dim=-1)
    gate_t = torch.sigmoid(r_gate_t)
    gamma_r = F.softplus(r_res)
    return tail_emb * (gate_t + gamma_r)
# ================= 5. 生成所有学生的候选打分 (非对称距离版) =================
print("⏳ 正在运行 CogRE 模型计算推荐基准得分 (同步 Leaky ReLU 逻辑)...")
uid_ex_scores = []
num_ex = len(Q)
all_ex_embs = torch.stack([get_ent_emb('ex' + str(qid)) for qid in range(num_ex)]).float()
rec_embedding = get_rel_emb('rec').float()

# 预先计算 t_projected
all_ex_proj = apply_cogre_tail(all_ex_embs, rec_embedding.unsqueeze(0)) 
gamma = 12.0
neg_slope = 0.2 # ⚠️ 必须和 model.py 里的 negative_slope 严格一致

# 为了节省内存，我们分批处理习题或直接用广播
for uid in uid_mlkc_dict.keys():
    uid_mlkc_keys = list(uid_mlkc_dict[uid].keys())
    mlkc_len = max(len(uid_mlkc_keys), 1)
    
    if len(uid_mlkc_keys) > 0:
        kc_embs = torch.stack([get_ent_emb(kc) for kc in uid_mlkc_keys]).float()
        mlkc_embs = torch.stack([get_rel_emb(uid_mlkc_dict[uid].get(kc, 'mlkc0.00'), 'mlkc0.00') for kc in uid_mlkc_keys]).float()
        pkc_embs = torch.stack([get_rel_emb(uid_pkc_dict[uid].get(kc, 'pkc0.00'), 'pkc0.00') for kc in uid_mlkc_keys]).float()

        # 计算 h_projected + r_m
        hr_mlkc = apply_cogre_head(kc_embs, mlkc_embs)
        hr_pkc = apply_cogre_head(kc_embs, pkc_embs)
        
        # -------------------------------------------------------------------
        # 🌟 核心修正：手动计算非对称距离，取代 torch.cdist
        # -------------------------------------------------------------------
        # 广播计算差值 [num_kc, num_ex, dim]
        # diff = t_proj - (h_proj + r_m)
        diff_mlkc = all_ex_proj.unsqueeze(0) - hr_mlkc.unsqueeze(1)
        diff_pkc = all_ex_proj.unsqueeze(0) - hr_pkc.unsqueeze(1)
        
        # 应用 Leaky ReLU 实现非对称惩罚
        diff_mlkc = F.leaky_relu(diff_mlkc, negative_slope=neg_slope)
        diff_pkc = F.leaky_relu(diff_pkc, negative_slope=neg_slope)
        
        # 计算范数 (如果训练是 p=1 这里也改 1.0)
        dist_mlkc = torch.norm(diff_mlkc, p=1.0, dim=-1) 
        dist_pkc = torch.norm(diff_pkc, p=1.0, dim=-1)
        
        fr1 = (gamma - dist_mlkc).sum(dim=0) + (gamma - dist_pkc).sum(dim=0)
    else:
        fr1 = torch.zeros(num_ex)

    # ---- 计算 fr2 (遗忘曲线也需同步) ----
    efr_embs = torch.stack([get_rel_emb(uid_exfr_dict[uid].get('ex'+str(qid), 'exfr0.00'), 'exfr0.00') for qid in range(num_ex)]).float()
    state_efr = apply_cogre_head(all_ex_embs, efr_embs)
    s_efr_rec = apply_cogre_head(state_efr, rec_embedding.unsqueeze(0))
    
    # 差值计算与非对称过滤
    diff_efr = all_ex_proj - s_efr_rec
    diff_efr = F.leaky_relu(diff_efr, negative_slope=neg_slope)
    dist_efr = torch.norm(diff_efr, p=1.0, dim=1) 
    
    fr2 = gamma - dist_efr

    O_sel = (fr1 / mlkc_len) + fr2 
    uid_ex_scores.append((uid, O_sel.tolist()))
'''

'''
# ================= 5. TripleRE 极简向量化评估 (Torch Native) =================
print("🚀 正在以 Torch 向量化模式运行 TripleRE 评估...")
uid_ex_scores = []
num_ex = len(Q)
# 1. 常数与全局投影
u, gamma = 2.0, 12.0  # u 需与训练参数一致
all_ex_embs = torch.stack([get_ent_emb(f'ex{i}') for i in range(num_ex)]).float()
rec_emb = get_rel_emb('rec').float()

# 2. 预切分 Rec 关系：r_h, r_t, r_m
r_h_rec, r_t_rec, r_m_rec = torch.chunk(rec_emb, 3, dim=-1)
# 预计算所有题目的尾部状态：t_proj = t * (r_t + u)
all_ex_proj = all_ex_embs * (r_t_rec + u) 

for uid in uid_mlkc_dict.keys():
    user_ml = uid_mlkc_dict[uid]
    mlkc_keys = list(user_ml.keys())
    
    if mlkc_keys:
        # --- 批量提取 Embedding ---
        h_embs = torch.stack([get_ent_emb(k) for k in mlkc_keys]).float()
        r_ml_embs = torch.stack([get_rel_emb(user_ml.get(k, 'mlkc0.00')) for k in mlkc_keys]).float()
        r_pk_embs = torch.stack([get_rel_emb(uid_pkc_dict[uid].get(k, 'pkc0.00')) for k in mlkc_keys]).float()

        # --- 核心逻辑函数化 ---
        def get_hr_triple(h, r_rel):
            rh, rt, rm = torch.chunk(r_rel, 3, dim=-1)
            # 学生状态：h * (rh + u) + rm
            state = h * (rh + u) + rm
            # 叠加推荐关系：state * (r_h_rec + u) + r_m_rec
            return state * (r_h_rec + u) + r_m_rec

        # 批量计算所有知识点对应的 hr
        hr_all = torch.cat([get_hr_triple(h_embs, r_ml_embs), get_hr_triple(h_embs, r_pk_embs)], dim=0)
        
        # 批量计算欧氏距离矩阵 [2*num_kc, num_ex] 并求和
        dist = torch.cdist(hr_all, all_ex_proj, p=2.0)
        fr1 = (gamma - dist).sum(dim=0)
    else:
        fr1 = torch.zeros(num_ex)

    # --- 遗忘曲线 fr2 向量化 ---
    r_fr_embs = torch.stack([get_rel_emb(uid_exfr_dict[uid].get(f'ex{i}', 'exfr0.00')) for i in range(num_ex)]).float()
    rfh, rft, rfm = torch.chunk(r_fr_embs, 3, dim=-1)
    
    # 历史习题变换：state_efr = e * (rfh + u) + rfm
    state_efr = all_ex_embs * (rfh + u) + rfm
    # 推荐变换后的 hr：s_efr_rec = state_efr * (r_h_rec + u) + r_m_rec
    s_efr_rec = state_efr * (r_h_rec + u) + r_m_rec
    
    # 自身偏移计算 (元素对齐计算，不需要 cdist)
    dist_efr = torch.norm(s_efr_rec - all_ex_proj, p=2.0, dim=1)
    fr2 = gamma - dist_efr

    # 最终汇总
    uid_ex_scores.append((uid, ((fr1 / max(len(mlkc_keys), 1)) + fr2).tolist()))

'''


import pickle

# ================= 5. 加载离线客观基准数据 (裁判端：仅大众题目难度) =================
print("-----------------------------------------------Load Offline Stats-----------------------------------------------")
try:
    with open("../data/algebra2005/raw_stats_akt.pkl", 'rb') as f:
        stats_data = pickle.load(f)
    # ⚠️ 现在只加载题目难度相关的离线基准
    exercise_df_dict = stats_data['exercise_df_dict']
    global_df_mean = stats_data['global_df_mean']
    print(f"成功加载离线统计基准，包含 {len(exercise_df_dict)} 道题目的客观难度 df_e！")
except Exception as e:
    print(f"无法读取 raw_stats_dkt.pkl，请确保已运行最新的预处理脚本！\n报错信息: {e}")
    exit()
    

# ================= 5.5 在线动态计算学生历史局部能力 (裁判端：个体细粒度能力) =================
print("------------------------------------------Load Online Student History-------------------------------------------")
csv_path = "../pyKT_example/output_final_akt.csv" # 确保路径与你的一致
student_kc_acc_dict = collections.defaultdict(lambda: collections.defaultdict(list))
global_correct = 0
global_total = 0

try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get('uid', '')).strip()
            concepts_str = str(row.get('concepts', '')).replace('nan', '')
            responses_str = str(row.get('responses', '')).replace('nan', '')
            
            if not concepts_str or not responses_str or not uid:
                continue
                
            concepts = concepts_str.split(',')
            responses = responses_str.split(',')
            
            # 对齐长度
            min_len = min(len(concepts), len(responses))
            for i in range(min_len):
                c_val = concepts[i].strip()
                r_val = responses[i].strip()
                if c_val and r_val:
                    try:
                        kc_id = int(float(c_val))
                        correct = int(float(r_val))
                        student_kc_acc_dict[uid][kc_id].append(correct)
                        global_correct += correct
                        global_total += 1
                    except ValueError:
                        pass
                        
    # 将列表聚合为均值正确率
    final_student_kc_acc = {}
    for uid, kc_data in student_kc_acc_dict.items():
        final_student_kc_acc[uid] = {}
        for kc_id, responses in kc_data.items():
            final_student_kc_acc[uid][kc_id] = sum(responses) / len(responses)
            
    global_H_mean = global_correct / global_total if global_total > 0 else 0.5
    
except Exception as e:

    final_student_kc_acc = {}
    global_H_mean = 0.5
    
import pickle
# ================= 6. 核心评估函数 (IRT + 几何平均) =================
def ACC_Dynamic_IRT(uid_mlkc_dict, uid_ex_scores, Q, n, student_kc_acc, exercise_df_dict, global_H_mean, global_df_mean):
    acc = []
    
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [idx_score[0] for idx_score in sorted_scores][:n]
        user_mlkc = uid_mlkc_dict.get(uid, {})
        
        diff_sum = 0
        for ex_id in uid_ex_score:
            
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            k_count = len(kc_list)
            
            # ----------------- 【裁判端：客观独立基准生成】 -----------------
            # 1. 题目难度 b_e (基于大众历史)
            df_e = exercise_df_dict.get(ex_id, global_df_mean)
            df_e_safe = np.clip(df_e, 0.05, 0.95)
            d = 1.0 - df_e_safe
            b_e = np.log(d / (1.0 - d))
            
            # 2. 细粒度学生能力 Theta_{i,e} (基于学生特定的考点历史)
            if k_count > 0:
                uid_str = str(uid)
                history = student_kc_acc.get(uid_str, {})
                local_obj_sum = sum([history.get(kc, global_H_mean) for kc in kc_list])
                H_ie_obj = local_obj_sum / k_count
            else:
                H_ie_obj = global_H_mean
                
            H_ie_safe = np.clip(H_ie_obj, 0.05, 0.95)
            theta_ie = np.log(H_ie_safe / (1.0 - H_ie_safe))
            
            # 3. 计算细粒度 IRT 客观胜率 (完美解耦！)
            sigma_ie = 1.0 / (1.0 + np.exp(-(theta_ie - b_e)))
            
            # ----------------- 【运动员端：模型主观预测】 -----------------
            ex_ml_product = 1.0
            for kc in kc_list:
                kc_key = 'kc' + str(kc)
                if kc_key in user_mlkc:
                    try:
                        # 只有这里用到了模型预测的 mlkc！
                        prob = float(user_mlkc[kc_key].replace('mlkc', ''))
                    except ValueError:
                        prob = global_H_mean
                else:
                    prob = global_H_mean 
                ex_ml_product *= prob
                
            D_ie = ex_ml_product ** (1.0 / k_count) if k_count > 0 else global_H_mean
            
            # ----------------- 【裁决：累加精准度】 -----------------
            diff_sum += 1.0 - np.abs(sigma_ie - D_ie)
            
        acc.append(diff_sum / n)
        
    return np.mean(acc), np.std(acc)


# Nov 保持原样，未做修改
def Nov(uid_kc_response, uid_ex_scores, Q, n):
    jaccsim = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        
        kc_response = set(uid_kc_response.get(uid, []))
        jaccard_similarity = 0
        for ex_id in uid_ex_score:
            rec_ex_kc_set = set()
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            rec_ex_kc_set.update(kc_list)
            
            intersection = len(kc_response.intersection(rec_ex_kc_set))
            union = len(kc_response.union(rec_ex_kc_set))
            if union > 0:
                jaccard_similarity += 1 - intersection / union
        jaccsim.append(jaccard_similarity / n)
    return np.mean(jaccsim), np.std(jaccsim)

def EMCC_Score_ZPD(uid_pkc_dict, uid_ex_scores, Q, n, student_kc_acc, exercise_df_dict, global_H_mean, global_df_mean, beta=1.0):
    
    emcc_list = []
    gamma_scale = 2.0  # Sigmoid 缩放因子，用于放大适切度与饱和度的差值效应
    
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        
        # 1. 选出得分最高的 n 道题
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_n_ex_ids = [idx_score[0] for idx_score in sorted_scores][:n]
        
        user_pkc = uid_pkc_dict.get(uid, {})
        uid_str = str(uid)
        history = student_kc_acc.get(uid_str, {})
        
        kc_alphas = collections.defaultdict(list)
        
        for ex_id in top_n_ex_ids:
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            k_count = len(kc_list)
            
            if k_count == 0:
                continue
                
            # --- 第一步：计算宏观题目适切度 (Exercise-Level Appropriateness) ---
            df_e = exercise_df_dict.get(ex_id, global_df_mean)
            df_e_safe = np.clip(df_e, 0.05, 0.95)
            b_norm = 1.0 - df_e_safe  # 客观难度归一化 (0~1)
            
            # 计算该题涉及知识点的平均基础能力
            local_obj_sum = sum([history.get(kc, global_H_mean) for kc in kc_list])
            H_ie = local_obj_sum / k_count
            H_ie_safe = np.clip(H_ie, 0.05, 0.95) # 综合能力归一化 (0~1)
            
            # 适切度：能力与难度的匹配程度 (越接近1越匹配，代表落入 ZPD)
            appropriateness = 1.0 - np.abs(H_ie_safe - b_norm)
            
            # --- 第二步：计算微观知识点习得率 (Concept-Level Alpha) ---
            for kc in kc_list:
                H_ik = history.get(kc, global_H_mean)
                H_ik_safe = np.clip(H_ik, 0.05, 0.95) # 知识点饱和度 (当前掌握度)
                
                logit = gamma_scale * (appropriateness - H_ik_safe)
                alpha_iek = 1.0 / (1.0 + np.exp(-logit))
                
                # 收集 alpha，用于后续的指数累加
                kc_alphas[kc].append(alpha_iek)
                
        # --- 第三 & 四步：计算 f_k(S) 并用 PKC 结算 ---
        total_expected_coverage = 0.0
        
        for kc, alphas in kc_alphas.items():
            kc_key = 'kc' + str(kc)
            pkc_val = 0.0
            if kc_key in user_pkc:
                try:
                    pkc_val = float(user_pkc[kc_key].replace('pkc', ''))
                except ValueError: 
                    pass
                      
            sum_alphas = np.sum(alphas)
            f_k_S = 1.0 - np.exp(-beta * sum_alphas)
            
            # 加权结算：需求权重(PKC) * 综合掌握度
            total_expected_coverage += pkc_val * f_k_S
            
        emcc_list.append(total_expected_coverage)
            
    return np.mean(emcc_list), np.std(emcc_list)

# ================= 7. 执行打印结果 =================
print("-----------------------------------------------Start calculating ACC-----------------------------------------------")
for n in [10, 20]:
    mean_acc, std_acc = ACC_Dynamic_IRT(
        uid_mlkc_dict, uid_ex_scores, Q, n, 
        final_student_kc_acc, exercise_df_dict, global_H_mean, global_df_mean
    )
    print(f"推荐列表长度 n = {n}, Mean ACC = {mean_acc:.4f}, Std ACC = {std_acc:.4f}")


print("-----------------------------------------------Start calculating NOV-----------------------------------------------")
all_uid_kc_response = {}
try:
    with open(f"{dict_path}/algebra2005_uid_kc_response.txt", 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            uid = line[0]
            correct_kc_response = [int(x) for x in line[1].split(',')]
            all_uid_kc_response[uid] = correct_kc_response
except Exception as e:
    print(f"无法读取 algebra2005_uid_kc_response.txt 文件: {e}")

test_uid_kc_response = {}
for uid in uid_mlkc_dict.keys():
    test_uid_kc_response[uid] = all_uid_kc_response.get(uid, [])

for n in [10, 20]:
    mean_nov, std_nov = Nov(test_uid_kc_response, uid_ex_scores, Q, n)
    print(f"推荐列表长度 n = {n}, Mean NOV = {mean_nov:.4f}, Std NOV = {std_nov:.4f}")

print("-----------------------------------------------Start calculating ZPD-EMCC@N-----------------------------------------------")
for n in [10, 20]:
    mean_emcc, std_emcc = EMCC_Score_ZPD(
        uid_pkc_dict, uid_ex_scores, Q, n, 
        final_student_kc_acc, exercise_df_dict, global_H_mean, global_df_mean, 
        beta=1.0
    )
    print(f"推荐列表长度 n = {n}, Mean ZPD-EMCC = {mean_emcc:.4f}, Std ZPD-EMCC = {std_emcc:.4f}")
