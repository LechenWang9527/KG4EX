import pandas as pd
import numpy as np
import pickle

def build_strict_stats_dict(csv_path, q_matrix_path):
    
    # ================= 1. 读取数据并拆分成列表 =================
    df = pd.read_csv(csv_path)
    
    # 🌟 移除 kc_pre 等学生能力字段，只保留计算题目难度必须的字段
    cols_to_explode = ['questions', 'concepts', 'responses']
    for col in cols_to_explode:
        # 注意这里去除了 nan 等空值的影响
        df[col] = df[col].astype(str).replace('nan', '').str.split(',')
        
    # ================= 🌟 核心修复：强制对齐列表长度 =================
    def align_lengths(row):
        # 找出当前行中，这3个列表的最短长度
        min_len = min(len(row[c]) for c in cols_to_explode if isinstance(row[c], list))
        # 按照最短长度，把所有列表强行截断对齐
        for c in cols_to_explode:
            if isinstance(row[c], list):
                row[c] = row[c][:min_len]
        return row
        
    # 应用这个修复逻辑
    df = df.apply(align_lengths, axis=1)

    # ================= 2. 安全炸开与类型清洗 =================
    df_exp = df.explode(cols_to_explode) # 现在绝对不会报错了！
    
    df_exp = df_exp[df_exp['concepts'].str.strip() != '']
    df_exp['concepts'] = pd.to_numeric(df_exp['concepts'], errors='coerce')
    df_exp['responses'] = pd.to_numeric(df_exp['responses'], errors='coerce') 
    
    # 丢掉那些无法转成数字的废数据
    df_exp = df_exp.dropna(subset=['concepts', 'responses'])
    
    # ================= 3. 计算客观题目难度 df_e =================
    # 先算：每个知识点在全网历史中的平均正确率
    df_k_dict = df_exp.groupby('concepts')['responses'].mean().to_dict()
    global_k_mean = float(df_exp['responses'].mean())
    
    Q = []
    with open(q_matrix_path, 'r') as file:
        for line in file:
            Q.append([int(x) for x in line.strip().split(',')])
            
    exercise_df_dict = {}
    
    for ex_idx, row in enumerate(Q):
        kc_indices = [idx for idx, val in enumerate(row) if val == 1]
        
        if len(kc_indices) > 0:
            k_probs = [df_k_dict.get(k, global_k_mean) for k in kc_indices]
            df_e = np.mean(k_probs) # 题目的客观难度 = 包含知识点正确率的平均
        else:
            df_e = global_k_mean
            
        exercise_df_dict[ex_idx] = float(df_e)
        
    print(f"✅ 成功映射了 {len(exercise_df_dict)} 道题目的客观难度 df_e。")
    
    # ================= 打印抽查：前 10 道题目 =================
    print("\n" + "="*50)
    print("前 10 道题目的客观难度 df_e :")
    for ex_idx, df_val in list(exercise_df_dict.items())[:10]:
        print(f"  题目 ID: ex{ex_idx:<4} | 客观难度 df_e = {df_val:.4f}")
    print("="*50 + "\n")
    
    # ================= 4. 打包封存 =================
    export_data = {
        "exercise_df_dict": exercise_df_dict,
        "global_df_mean": global_k_mean
        # 学生全局能力被彻底移除，将在后续在线评估中动态计算！
    }
    
    with open("raw_stats_akt.pkl", 'wb') as f:
        pickle.dump(export_data, f)
    print("🎉 数据已精简并封存至 [raw_stats_akt.pkl] (仅包含客观难度基准)")


if __name__ == "__main__":
    # 请确保路径正确
    csv_file_path = "../../pyKT_example/output_final_akt_fpkc_h1_g08.csv" 
    q_matrix_path = "Q.txt"    
    build_strict_stats_dict(csv_file_path, q_matrix_path)