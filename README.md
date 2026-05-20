# 

这份代码主要做两件事：

第一，用训练好的嵌入向量给每个学生的候选习题打分，得到推荐列表。

第二，在推荐列表上算三个指标：`ACC@N`、`NOV@N`、`ZPD-EMCC@N`。我们现在重点看 `ACC` 和 `EMCC`。

这里的 `N` 一般取 `10` 和 `20`，所以最后会打印 `ACC@10 / ACC@20 / EMCC@10 / EMCC@20`。

## 模型和训练这块

这一块主要对应 `model.py` 和 `run.py`。简单说，训练阶段是在学一套实体向量和关系向量；测试阶段再用这些向量给每个学生的候选习题打分。

这次训练用的是 `CogRE` 模型，不是普通的 `TransE`。它还是知识图谱嵌入模型的思路：

```text
实体：学生、知识点、习题等
关系：mlkc、pkc、exfr、rec 等
目标：让真实三元组的分数更高，让负采样三元组的分数更低
```

### 1）CogRE 的模型结构

训练时会初始化两类参数：

```python
entity_embedding   # 实体向量
relation_embedding # 关系向量
```

这次指令里设置了：

```text
-d 500
```

所以实体向量维度是：

$$
entity\_dim = 500
$$

CogRE 会把一个关系向量拆成 4 段，所以关系向量维度是：

$$
relation\_dim = 4 \times 500 = 2000
$$

也就是：

$$
r = [r_h, r_t, r_m, r_{res}]
$$

这里可以这么理解：

```text
r_h：控制头实体怎么被关系筛选
r_t：控制尾实体怎么被关系筛选
r_m：关系带来的平移项
r_res：关系残差，用来补充门控强度
```

在模型里，头实体和尾实体不是直接相减，而是先做关系门控投影：

$$
h' = h \odot (\sigma(r_h) + 0.2\sigma(r_{res}))
$$

$$
t' = t \odot (\sigma(r_t) + 0.2\sigma(r_{res}))
$$

然后计算：

$$
diff = t' - (h' + r_m)
$$

CogRE 这里用了一个非对称惩罚，不是普通的 L1：

$$
penalty_j =
\begin{cases}
|diff_j|, & diff_j > 0 \\
\alpha |diff_j|, & diff_j \le 0
\end{cases}
$$

代码里：

$$
\alpha = 0.4
$$

最后三元组分数是：

$$
score(h,r,t)=\gamma-\sum_j penalty_j
$$

这次训练没有手动传 `-g`，所以 `gamma` 用的是 `run.py` 里的默认值：

```text
-gamma = 12.0
```

直观理解就是：真实三元组希望距离小、分数高；错误三元组希望距离大、分数低。

### 2）训练时怎么构造正负样本

训练数据来自：

```python
akt_fpkc_h1_g08_triples.txt
```

每条真实三元组是正样本。负样本通过随机替换头实体或尾实体得到：

```text
head-batch：替换头实体
tail-batch：替换尾实体
```

训练时会在 `head-batch` 和 `tail-batch` 之间交替取数据，这样不会只学会一边的替换。

这次训练指令里：

```text
-b 512
-n 128
```

意思是：

```text
batch size = 512
每个正样本配 128 个负样本
```

### 3）损失函数和优化器

优化器用的是 Adam：

```python
optimizer = torch.optim.Adam(..., lr=0.001)
```

因为训练指令里写的是：

```text
-lr 0.001
```

训练目标大概是：

```text
正样本分数越高越好
负样本分数越低越好
```

代码里对正样本用了：

$$
-\log \sigma(score_{pos})
$$

对负样本用了：

$$
-\log \sigma(-score_{neg})
$$

因为指令里加了：

```text
-adv
```

所以负样本不是简单平均，而是用了自对抗负采样。也就是说，模型会更关注那些“看起来更像真的”的负样本，因为这些负样本更难分。

代码形式大概是：

```python
negative_score = (
    F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
    * F.logsigmoid(-negative_score)
).sum(dim=1)
```

最后正样本损失和负样本损失取平均：

$$
loss = \frac{loss_{pos}+loss_{neg}}{2}
$$

这次训练里：

```text
-r 0.00
```

所以没有额外正则项。

### 4）学习率和训练步数

这次最大训练步数是：

```text
--max_steps 30000
```

如果没有手动设置 `warm_up_steps`，代码默认：

$$
warm\_up\_steps = \frac{max\_steps}{2} = 15000
$$

也就是说，训练到 15000 步左右时，学习率会除以 10：

```text
0.001 -> 0.0001
```

模型会定期保存 checkpoint。代码默认每 100 步保存一次：

```text
--save_checkpoint_steps 100
```

### 5）随机种子会不会带来偶然性

会的。当前这份 `run.py` 里没有固定随机种子，所以每次重新训练，即使用同一条训练指令，结果也可能有一点差异。

主要随机性来自这些地方：

```text
实体向量和关系向量的随机初始化
DataLoader 的 shuffle
负采样时随机选实体
GPU 上某些算子的非确定性
```

所以严格来说：

```text
同一份代码 + 同一份数据 + 同一条训练指令
不一定每次训练结果完全一样
```

如果只是加载同一个已经训练好的 checkpoint 去测试，结果一般会稳定；但如果是重新训练模型，结果就可能波动。

比较严谨的做法是跑多个 seed，比如 3 个或 5 个，然后汇报：

$$
mean \pm std
$$

如果想先固定随机种子，可以在 `run.py` 里加类似这样的函数：

```python
def set_seed(seed=2024):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

然后在 `main(parse_args())` 前调用：

```python
set_seed(2024)
```

不过就算固定 seed，GPU 环境、库版本、多线程设置不一样，也可能有很小差异。所以论文或者报告里最好还是用多次实验的均值和标准差。

### 6）本次训练指令

本次训练直接运行：

```bash
python run.py --do_train --cuda \
  --data_path ../data/algebra2005 \
  --save_path ./models/algebra2005/cog6 \
  --model CogRE \
  -b 512 \
  -n 128 \
  -d 500 \
  -r 0.00 \
  -lr 0.001 \
  -cpu 20 \
  -3u 1.0 \
  --max_steps 30000 \
  -adv
```

各参数对应的意思：

```text
--do_train：训练模式
--cuda：用 GPU 训练
--data_path：数据集路径
--save_path：模型保存路径
--model CogRE：使用 CogRE 模型
-b 512：batch size 是 512
-n 128：每个正样本采 128 个负样本
-d 500：实体向量维度是 500
-r 0.00：不加正则
-lr 0.001：学习率是 0.001
-cpu 20：DataLoader 使用的 CPU 线程参数
-3u 1.0：TripleRE 用的参数，这次 CogRE 基本用不上
--max_steps 30000：训练 30000 步
-adv：开启自对抗负采样
```

需要注意一点：`run.py` 里训练三元组文件目前是写死的：

```python
akt_fpkc_h1_g08_triples.txt
```

如果要训练别的版本，比如 `h5_g08`、`h5_g09`，这里也要一起改，不然保存路径换了，但训练数据其实还是旧的。

---

---

## 1. 整体数据流

`test_TransE.py` 的运行顺序大概是这样：

```text
加载模型向量
    ↓
读取 Q 矩阵、实体字典、关系字典
    ↓
读取测试三元组，整理成每个学生的 mlkc / pkc / exfr 信息
    ↓
用 CogRE 打分函数给每个学生的全部候选习题打分
    ↓
按分数排序，取前 N 道题
    ↓
计算 ACC@N、NOV@N、ZPD-EMCC@N
```

几个重要的数据结构：

```python
uid_mlkc_dict[uid][kc] = "mlkc0.xx"
```

表示模型认为学生 `uid` 对知识点 `kc` 的当前掌握程度。

```python
uid_pkc_dict[uid][kc] = "pkc0.xx"
```

表示学生 `uid` 对知识点 `kc` 的后续学习需求。这个值在 `EMCC` 里会作为权重。

```python
uid_exfr_dict[uid][ex] = "exfr0.xx"
```

表示学生 `uid` 对历史习题 `ex` 的遗忘状态。

```python
Q[exercise_id]
```

是习题到知识点的映射。也就是一道题包含哪些知识点。

```python
exercise_df_dict[exercise_id]
```

是题目的历史正确率。正确率越低，题目越难。

```python
student_kc_acc[uid][kc]
```

是学生在某个知识点上的历史正确率，用来估计学生真实能力。

---

## 2. 候选习题怎么打分

打分这一步不是评价指标本身，它只是先给每个学生生成一个推荐列表。后面的 `ACC` 和 `EMCC` 都是在这个推荐列表上算的。

CogRE 里把关系向量切成四段：

$$
r = [r_h, r_t, r_m, r_{res}]
$$

头实体投影：

$$
\phi_h(x,r)=x\odot(\sigma(r_h)+0.2\sigma(r_{res}))+r_m
$$

尾实体投影：

$$
\phi_t(x,r)=x\odot(\sigma(r_t)+0.2\sigma(r_{res}))
$$

非对称 L1 距离：

$$
d(a,b)=\sum_j
\begin{cases}
|a_j-b_j|, & a_j-b_j>0 \\
\alpha |a_j-b_j|, & a_j-b_j\le 0
\end{cases}
$$

代码里：

```python
COGRE_ALPHA = 0.4
GAMMA = 12.0

def calc_alpha_l1(diff_tensor, alpha=COGRE_ALPHA):
    abs_diff = torch.abs(diff_tensor)
    penalty = torch.where(diff_tensor > 0, abs_diff, alpha * abs_diff)
    return torch.sum(penalty, dim=-1)
```

对某个学生 `u` 和某道候选题 `e`，最终分数由两部分组成：

```text
fr1：知识点掌握度和学习需求带来的推荐分
fr2：历史习题遗忘状态带来的推荐分
```

公式可以写成：

$$
score(u,e)=\frac{fr1(u,e)}{|C_u|}+fr2(u,e)
$$

其中 `C_u` 是学生已有记录里的知识点集合。

代码核心逻辑是：

```python
hr_mlkc = apply_cogre_head(concept_embeddings, mlkc_embeddings)
hr_pkc = apply_cogre_head(concept_embeddings, pkc_embeddings)

all_exercise_projection = apply_cogre_tail(all_exercise_embeddings, rec_embedding)

dist_mlkc = calc_alpha_l1(hr_mlkc.unsqueeze(1) - all_exercise_projection.unsqueeze(0))
dist_pkc = calc_alpha_l1(hr_pkc.unsqueeze(1) - all_exercise_projection.unsqueeze(0))

fr1 = (GAMMA - dist_mlkc).sum(dim=0) + (GAMMA - dist_pkc).sum(dim=0)

state_efr = apply_cogre_head(all_exercise_embeddings, exfr_embeddings)
state_efr_rec = apply_cogre_head(state_efr, rec_embedding)

dist_efr = calc_alpha_l1(state_efr_rec - all_exercise_projection)
fr2 = GAMMA - dist_efr

final_scores = (fr1 / concept_count) + fr2
```

得到 `final_scores` 后，对每个学生按分数从高到低排序，取前 `N` 道题。

---

## 3. ACC@N 是怎么算的

这里的 `ACC` 不是传统二分类准确率。它的意思是：

> 推荐出来的题目，模型预测的掌握概率，和一个比较客观的 IRT 估计概率有多接近。

对学生 `u` 推荐出的前 `N` 道题记为：

$$
S_u^N
$$

对其中一道题 `e`，它涉及的知识点集合是：

$$
K_e
$$

### 3.1 先算题目难度

代码里 `df_e` 是题目历史正确率，所以题目难度定义为：

$$
d_e = 1 - df_e
$$

然后转成 IRT 里的难度参数：

$$
b_e=\log\frac{d_e}{1-d_e}
$$

代码：

```python
df_e = exercise_df_dict.get(exercise_id, global_df_mean)
df_e_safe = np.clip(df_e, 0.05, 0.95)

difficulty = 1.0 - df_e_safe
b_e = np.log(difficulty / (1.0 - difficulty))
```

### 3.2 再算学生对这道题的客观能力

如果一道题包含多个知识点，就取学生在这些知识点上的历史正确率平均值：

$$
H_{u,e}=\frac{1}{|K_e|}\sum_{k\in K_e}H_{u,k}
$$

再转成能力参数：

$$
\theta_{u,e}=\log\frac{H_{u,e}}{1-H_{u,e}}
$$

代码：

```python
history = student_kc_acc.get(str(uid), {})
student_ability = sum(
    history.get(concept_id, global_history_mean)
    for concept_id in concept_ids
) / concept_count

ability_safe = np.clip(student_ability, 0.05, 0.95)
theta_ie = np.log(ability_safe / (1.0 - ability_safe))
```

### 3.3 得到 IRT 客观成功概率

$$
P^{obj}_{u,e}=\sigma(\theta_{u,e}-b_e)
$$

也就是：

```python
objective_success_prob = 1.0 / (1.0 + np.exp(-(theta_ie - b_e)))
```

### 3.4 再算模型自己的掌握预测

模型预测来自 `mlkc`。如果一道题有多个知识点，就用几何平均：

$$
P^{model}_{u,e}=\left(\prod_{k\in K_e}MLKC_{u,k}\right)^{\frac{1}{|K_e|}}
$$

代码：

```python
model_mastery_product = 1.0
for concept_id in concept_ids:
    concept_key = f"kc{concept_id}"
    prob = float(user_mlkc[concept_key].replace("mlkc", ""))
    model_mastery_product *= prob

predicted_mastery = model_mastery_product ** (1.0 / concept_count)
```

### 3.5 最后算 ACC@N

每道题的得分是：

$$
1-|P^{obj}_{u,e}-P^{model}_{u,e}|
$$

越接近 1，说明模型预测越贴近客观估计。

所以单个学生的 `ACC@N` 是：

$$
ACC_u@N=\frac{1}{N}\sum_{e\in S_u^N}\left(1-|P^{obj}_{u,e}-P^{model}_{u,e}|\right)
$$

最后对所有测试学生求均值和标准差：

$$
ACC@N=mean_u(ACC_u@N)
$$

代码：

```python
diff_sum += 1.0 - np.abs(objective_success_prob - predicted_mastery)
acc_values.append(diff_sum / n)

return np.mean(acc_values), np.std(acc_values)
```

---

## 4. ZPD-EMCC@N 是怎么算的

`EMCC` 关注的是：

> 推荐列表能不能覆盖学生真正需要补的知识点，而且这些题是不是适合当前学生做。

它不是只看推荐题目多不多，而是看推荐列表对学生后续学习的预期收益。

对学生 `u`，推荐列表还是：

$$
S_u^N
$$

最终公式是：

$$
EMCC_u@N=\sum_{k\in K(S_u^N)}PKC_{u,k}\cdot f_{u,k}(S_u^N)
$$

其中：

```text
PKC_{u,k}：学生对知识点 k 的后续学习需求
f_{u,k}(S)：推荐列表 S 对知识点 k 的联合覆盖效果
```

### 4.1 先算题目适切度

题目难度仍然来自历史正确率：

$$
b^{norm}_e=1-df_e
$$

学生对这道题相关知识点的综合能力是：

$$
H_{u,e}=\frac{1}{|K_e|}\sum_{k\in K_e}H_{u,k}
$$

题目适切度：

$$
Suit_{u,e}=1-|H_{u,e}-b^{norm}_e|
$$

意思很简单：学生能力和题目难度越接近，这道题越适合他。

代码：

```python
normalized_difficulty = 1.0 - df_e_safe

exercise_ability = sum(
    history.get(concept_id, global_history_mean)
    for concept_id in concept_ids
) / concept_count

appropriateness = 1.0 - np.abs(exercise_ability_safe - normalized_difficulty)
```

### 4.2 再算单题对知识点的习得贡献

对题目 `e` 里的某个知识点 `k`，学生当前掌握度是：

$$
H_{u,k}
$$

单题习得贡献定义为：

$$
\alpha_{u,e,k}=\sigma(\gamma_s(Suit_{u,e}-H_{u,k}))
$$

代码里：

$$
\gamma_s=2.0
$$

直观理解：

```text
题目越适合学生，alpha 越大；
学生已经很熟的知识点，alpha 会被压低；
学生还没掌握、但题目难度又比较合适，alpha 会更高，这个要不要暂时没决定
```

代码：

```python
gamma_scale = 2.0

concept_mastery = history.get(concept_id, global_history_mean)
concept_mastery_safe = np.clip(concept_mastery, 0.05, 0.95)

logit = gamma_scale * (appropriateness - concept_mastery_safe)
alpha_iek = 1.0 / (1.0 + np.exp(-logit))

concept_alphas[concept_id].append(alpha_iek)
```

### 4.3 多道题覆盖同一个知识点时，做边际收益递减

如果推荐列表里有多道题都覆盖知识点 `k`，不是简单相加，而是用指数函数压一下：

$$
f_{u,k}(S)=1-\exp\left(-\beta\sum_{e\in S_k}\alpha_{u,e,k}\right)
$$

代码里：

$$
\beta=1.0
$$

这样做的原因是：同一个知识点推荐太多题，收益会逐渐饱和，不会无限增加。

代码：

```python
f_k_s = 1.0 - np.exp(-beta * np.sum(alphas))
```

### 4.4 最后乘上 PKC，得到 EMCC

`PKC` 是学生对某个知识点的后续学习需求。需求越高，这个知识点的覆盖越重要。

$$
EMCC_u@N=\sum_k PKC_{u,k}\cdot f_{u,k}(S_u^N)
$$

代码：

```python
expected_coverage = 0.0

for concept_id, alphas in concept_alphas.items():
    concept_key = f"kc{concept_id}"
    pkc_value = float(user_pkc[concept_key].replace("pkc", ""))

    f_k_s = 1.0 - np.exp(-beta * np.sum(alphas))
    expected_coverage += pkc_value * f_k_s

emcc_values.append(expected_coverage)

return np.mean(emcc_values), np.std(emcc_values)
```

---

## 5. ACC 和 EMCC 的区别

`ACC` 看的是：

```text
模型对学生掌握度的预测，和客观 IRT 估计是否接近。
```

所以 `ACC` 越高，说明推荐列表里的题，模型对学生能不能做对的判断越稳。

`EMCC` 看的是：

```text
推荐列表能不能覆盖学生未来真正需要补的知识点，并且题目难度是否合适。
```

所以 `EMCC` 越高，说明推荐列表更有学习收益，更像是在帮学生补短板。

简单说：

```text
ACC：推荐是否“判断准确”
EMCC：推荐是否“有学习价值”
```

这两个指标不一定同时升高。比如推荐很多很简单的题，`ACC` 可能不错，但 `EMCC` 不一定高；推荐更有挑战、刚好卡在学生薄弱点上的题，`EMCC` 可能更高，但 `ACC` 可能会波动。

---

## 6. 现在代码里打印结果的格式，test_TransE3.py 里的是改进版的test,test_TransE.py是老版

现在打印会比较简洁：

```text
Loading data...
Scoring candidates...
Candidate scoring complete.
Calculating ACC...
n=10 | ACC mean=0.xxxx, std=0.xxxx
n=20 | ACC mean=0.xxxx, std=0.xxxx
Calculating NOV...
n=10 | NOV mean=0.xxxx, std=0.xxxx
n=20 | NOV mean=0.xxxx, std=0.xxxx
Calculating ZPD-EMCC...
n=10 | ZPD-EMCC mean=0.xxxx, std=0.xxxx
n=20 | ZPD-EMCC mean=0.xxxx, std=0.xxxx
```

不会再打一堆横线和符号。

---

## 7. 跑代码前要确认的路径

`test_TransE.py` 顶部有这些路径：

```python
DATA_PATH = Path("../data/algebra2005")
EMBEDDING_PATH = Path("./models/algebra2005/cog_h1_g08")
TEST_TRIPLES_PATH = DATA_PATH / "akt_fpkc_h1_g08_test_triples.txt"
OFFLINE_STATS_PATH = DATA_PATH / "raw_stats_akt.pkl"
STUDENT_HISTORY_PATH = Path("../pyKT_example/output_final_akt.csv")
UID_KC_RESPONSE_PATH = DATA_PATH / "algebra2005_uid_kc_response.txt"
```

换实验的时候，一般要改这两个：

```python
EMBEDDING_PATH = Path("./models/algebra2005/你的模型目录")
TEST_TRIPLES_PATH = DATA_PATH / "你的测试三元组文件.txt"
```

比如从 `h1_g08` 换成 `h5_g09`，就要保证模型目录和测试三元组文件是一套的，不然结果会对不上。
