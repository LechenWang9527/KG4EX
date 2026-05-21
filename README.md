# KG4EX-CogRE: 基于知识图谱嵌入的可解释习题推荐

本项目面向个性化习题推荐任务。整体思路是：先从学生历史作答序列中提取学习特征，再把学生、知识点和习题组织成知识图谱，随后使用知识图谱嵌入模型学习实体和关系向量，最后根据嵌入空间中的推理得分为每个学生生成推荐习题列表。

和普通的习题推荐方法相比，本项目关注两件事。第一，推荐结果本身要尽量准确，模型要能判断学生是否适合做某道题。第二，推荐结果要有学习收益，也就是推荐列表要覆盖学生后续真正需要补充的知识点，并且题目难度要落在学生当前能力附近。

因此，当前版本主要使用两个核心评价指标：`ACC@N` 和 `ZPD-EMCC@N`。其中，`ACC@N` 衡量模型预测掌握概率与客观 IRT 估计概率的一致性；`ZPD-EMCC@N` 衡量推荐列表对学生未来学习需求的期望覆盖收益。旧版的新颖性指标 `NOV@N` 在当前版本中不再作为主要指标。

---

## 1. 方法概览

整个流程可以概括为五个阶段：

```text
历史学习交互数据
    ↓
学生特征提取：MLKC / PKC / EFR
    ↓
知识图谱构建：student / concept / exercise / relation
    ↓
知识图谱嵌入训练：CogRE / TripleRE / TransE-ADV
    ↓
候选习题打分与评价：ACC@N / ZPD-EMCC@N
```

前半部分的特征提取和知识图谱构建沿用了 KG4EER 的基本思想：从学生历史交互中提取知识点掌握程度、后续知识点出现概率和遗忘状态，再把这些特征转化成知识图谱中的显式关系。当前版本的主要改动集中在嵌入模型、候选习题打分方式和评价指标上。

---

## 2. 数据与符号说明

设学生集合为：

$$
S=\{s_1,s_2,\ldots,s_{N_s}\}
$$

习题集合为：

$$
E=\{e_1,e_2,\ldots,e_{N_e}\}
$$

知识点集合为：

$$
K=\{k_1,k_2,\ldots,k_{N_k}\}
$$

`Q.txt` 是习题和知识点的关联矩阵：

$$
Q_{e,k}=\begin{cases}
1, & \text{习题 } e \text{ 涉及知识点 } k \\
0, & \text{否则}
\end{cases}
$$

在代码中，一道习题 `ex_i` 涉及的知识点集合记为：

$$
K_e=\{k \mid Q_{e,k}=1\}
$$

本项目中的主要实体包括：

```text
uid*  : 学生实体
kc*   : 知识点实体
ex*   : 习题实体
```

主要关系包括：

```text
mlkc* : 学生对知识点的当前掌握程度
pkc*  : 学生后续学习中对知识点的需求强度
exfr* : 学生对历史习题相关内容的遗忘状态
rec   : 学生和习题之间的推荐关系
```

三元组文件通常采用如下格式：

```text
head_entity    relation    tail_entity
```

例如：

```text
kc12    mlkc0.73    uid8
kc12    pkc0.41     uid8
ex233   exfr0.62    uid8
uid8    rec         ex233
```

这些三元组会被 `generate_dict.py` 转换成实体字典和关系字典，并进一步用于图谱嵌入训练。

---

## 3. 学生特征提取

特征提取模块的目标是把学生历史作答序列转化为可进入知识图谱的结构化关系。本项目主要使用三类学生学习特征。

### 3.1 MLKC：知识点掌握程度

`MLKC` 表示学生在当前时刻对某个知识点的掌握程度。对学生 $u$ 和知识点 $k$，记为：

$$
MLKC_{u,k}\in[0,1]
$$

数值越大，说明模型认为学生越可能掌握该知识点。该特征通常由知识追踪模型或序列模型根据学生历史作答记录得到。

在知识图谱中，它会被离散化成关系形式，例如：

```text
kc12    mlkc0.73    uid8
```

含义是：学生 `uid8` 对知识点 `kc12` 的掌握程度约为 `0.73`。

### 3.2 PKC：后续知识点需求概率

`PKC` 表示某个知识点在学生后续学习中出现或被需要的概率。对学生 $u$ 和知识点 $k$，记为：

$$
PKC_{u,k}\in[0,1]
$$

它不是简单表示学生是否已经掌握某个知识点，而是表示这个知识点对学生后续学习任务的重要程度。在本项目的 `ZPD-EMCC` 指标中，`PKC` 会作为知识点学习收益的权重。

在知识图谱中，对应关系例如：

```text
kc12    pkc0.41    uid8
```

含义是：知识点 `kc12` 对学生 `uid8` 后续学习的需求强度约为 `0.41`。

### 3.3 EFR / EXFR：遗忘状态

`EFR` 或 `EXFR` 表示学生对历史习题相关知识的遗忘状态。遗忘建模的直观依据是：学生并不会永久保持同一水平的知识掌握度，随着时间推移，已经学过的内容会产生遗忘。

对学生 $u$ 和习题 $e$，遗忘状态可记为：

$$
EXFR_{u,e}\in[0,1]
$$

在知识图谱中，对应关系例如：

```text
ex233    exfr0.62    uid8
```

含义是：学生 `uid8` 对习题 `ex233` 相关内容的遗忘程度约为 `0.62`。

---

## 4. 知识图谱构建

知识图谱用于把学生、知识点和习题放到统一的结构中。当前项目中，一条知识图谱三元组可以写成：

$$
(h,r,t)
$$

其中 $h$ 是头实体，$r$ 是关系，$t$ 是尾实体。

本项目的图谱主要包含四类语义关系：

| 关系类型 | 示例 | 含义 |
|---|---|---|
| `mlkc` | `kc12 mlkc0.73 uid8` | 学生对知识点的掌握程度 |
| `pkc` | `kc12 pkc0.41 uid8` | 学生后续学习对知识点的需求 |
| `exfr` | `ex233 exfr0.62 uid8` | 学生对历史习题相关内容的遗忘状态 |
| `rec` | `uid8 rec ex233` | 学生和候选习题之间的推荐关系 |

图谱构建完成后，所有实体和关系会被映射成整数 ID：

```text
entities.dict
relations.dict
```

训练阶段的输入是 ID 化后的三元组。模型训练完成后，会保存：

```text
entity_embedding.npy
relation_embedding.npy
checkpoint
config.json
```

测试阶段再加载这些嵌入向量，为每个学生计算候选习题分数。

---

## 5. CogRE 嵌入模型

当前版本的核心模型是 `CogRE`。它仍然属于知识图谱嵌入模型，但相比普通 `TransE`，它不再简单使用：

$$
h+r\approx t
$$

而是对头实体和尾实体分别做关系门控投影，再计算非对称距离。

### 5.1 参数维度

训练时初始化两类参数：

```python
entity_embedding
relation_embedding
```

若实体向量维度为：

$$
d=500
$$

则实体向量维度为：

$$
entity\_dim=d
$$

CogRE 将每个关系向量划分为四段：

$$
r=[r_h,r_t,r_m,r_{res}]
$$

所以关系向量维度为：

$$
relation\_dim=4d
$$

当 $d=500$ 时，关系向量维度为 `2000`。

### 5.2 关系门控投影

对三元组 $(h,r,t)$，先将关系向量拆分为：

$$
r_h,r_t,r_m,r_{res}
$$

头实体投影为：

$$
h'=h\odot\left(\sigma(r_h)+0.2\sigma(r_{res})\right)
$$

尾实体投影为：

$$
t'=t\odot\left(\sigma(r_t)+0.2\sigma(r_{res})\right)
$$

其中 $\odot$ 表示逐元素乘法，$\sigma(\cdot)$ 表示 Sigmoid 函数。

随后计算方向差：

$$
diff=t'-(h'+r_m)
$$

这个方向必须和测试阶段保持一致。因为 CogRE 使用的是非对称距离，方向反过来会改变排序结果。

### 5.3 非对称 L1 惩罚

CogRE 使用非对称 L1 惩罚：

$$
penalty_j=
\begin{cases}
|diff_j|, & diff_j>0 \\
\alpha |diff_j|, & diff_j\le 0
\end{cases}
$$

当前代码中：

$$
\alpha=0.4
$$

最终三元组得分为：

$$
score(h,r,t)=\gamma-\sum_j penalty_j
$$

其中 $\gamma$ 是 margin 参数，默认值为：

$$
\gamma=12.0
$$

分数越高，说明三元组越可信。

---

## 6. 训练目标

训练数据中的真实三元组作为正样本。负样本通过随机替换头实体或尾实体得到：

```text
head-batch: 固定 relation 和 tail，替换 head
tail-batch: 固定 head 和 relation，替换 tail
```

训练过程中，`BidirectionalOneShotIterator` 会在 head-batch 和 tail-batch 之间交替取 batch，使模型同时学习两种负采样方向。

### 6.1 正负样本损失

对正样本分数 $s^+$，使用：

$$
L_{pos}=-\log\sigma(s^+)
$$

对负样本分数 $s^-$，使用：

$$
L_{neg}=-\log\sigma(-s^-)
$$

如果开启自对抗负采样，模型会给更难的负样本更高权重。负样本权重为：

$$
w_i=\frac{\exp(\alpha_s s_i^-)}{\sum_j \exp(\alpha_s s_j^-)}
$$

其中 $\alpha_s$ 是 adversarial temperature。最终负样本损失为加权形式：

$$
L_{neg}=-\sum_i w_i\log\sigma(-s_i^-)
$$

总损失为：

$$
L=\frac{L_{pos}+L_{neg}}{2}
$$

当前版本已经恢复原始 KGE 损失函数，没有加入额外的关系平滑损失或三元组加权损失。

### 6.2 优化器和学习率

训练使用 Adam 优化器：

```python
optimizer = torch.optim.Adam(..., lr=args.learning_rate)
```

常用训练指令如下：

```bash
python run.py --do_train --cuda \
  --data_path ../data/algebra2005 \
  --save_path ./models/algebra2005/cog_h5_g09 \
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

参数说明：

| 参数 | 含义 |
|---|---|
| `--do_train` | 开启训练模式 |
| `--cuda` | 使用 GPU |
| `--data_path` | 数据目录 |
| `--save_path` | 模型保存目录 |
| `--model CogRE` | 使用 CogRE 模型 |
| `-b 512` | batch size |
| `-n 128` | 每个正样本对应的负样本数量 |
| `-d 500` | 实体向量维度 |
| `-r 0.00` | 正则系数 |
| `-lr 0.001` | 学习率 |
| `-cpu 20` | DataLoader 使用的 CPU worker 参数 |
| `-3u 1.0` | TripleRE 参数，CogRE 下基本不参与计算 |
| `--max_steps 30000` | 最大训练步数 |
| `-adv` | 开启自对抗负采样 |

如果希望减少训练过程中的磁盘写入，可以增大保存间隔：

```bash
--save_checkpoint_steps 5000 --log_steps 500
```

---

## 7. 候选习题打分

训练完成后，测试脚本会加载：

```text
entity_embedding.npy
relation_embedding.npy
entities.dict
relations.dict
Q.txt
test triples
```

对每个学生，测试脚本先读取其 `mlkc`、`pkc` 和 `exfr` 信息，然后对所有候选习题计算分数。

### 7.1 CogRE 推荐分数

对学生 $u$ 和候选习题 $e$，最终得分由两部分组成：

$$
Score(u,e)=\frac{fr_1(u,e)}{|C_u|}+fr_2(u,e)
$$

其中 $C_u$ 是学生在测试三元组中出现过的知识点集合。

第一部分 $fr_1$ 来自知识点掌握程度和后续学习需求：

$$
fr_1(u,e)=\sum_{k\in C_u}\left(score(k,MLKC_{u,k},e)+score(k,PKC_{u,k},e)\right)
$$

第二部分 $fr_2$ 来自遗忘状态：

$$
fr_2(u,e)=score(e,EXFR_{u,e},e)
$$

在 CogRE 中，测试阶段必须使用和训练阶段一致的方向：

$$
diff=t'-(h'+r_m)
$$

对于候选习题得分，代码中对应的核心方向是：

```python
diff_mlkc = all_ex_proj.unsqueeze(0) - hr_mlkc.unsqueeze(1)
diff_pkc = all_ex_proj.unsqueeze(0) - hr_pkc.unsqueeze(1)
diff_efr = all_ex_proj - state_efr_rec
```

如果方向写反，候选题排序会发生改变，最终 `ACC@N` 和 `ZPD-EMCC@N` 也会改变。

---

## 8. 评价指标

当前版本重点使用两个指标：

```text
ACC@N
ZPD-EMCC@N
```

其中 $N$ 通常取 `10` 和 `20`。

### 8.1 ACC@N

`ACC@N` 衡量推荐列表中，模型预测的学生掌握概率是否接近一个客观估计的成功概率。这里的 `ACC` 不是传统分类准确率，而是推荐列表上的预测一致性指标。

对学生 $u$，模型推荐的前 $N$ 道题记为：

$$
S_u^N
$$

对其中一道题 $e$，其知识点集合为 $K_e$。

#### 8.1.1 题目客观难度

离线统计文件 `raw_stats_akt.pkl` 中保存了：

```python
exercise_df_dict
```

其中 $df_e$ 表示习题 $e$ 的客观正确率估计。当前预处理逻辑中，$df_e$ 主要由该题涉及知识点的历史正确率通过 Q 矩阵聚合得到：

$$
df_e=\frac{1}{|K_e|}\sum_{k\in K_e}df_k
$$

其中 $df_k$ 是知识点 $k$ 的历史正确率。如果某个知识点缺少历史统计，则使用全局平均正确率作为兜底。

题目难度定义为：

$$
d_e=1-df_e
$$

为了避免 logit 发散，代码会进行裁剪：

$$
df_e\leftarrow clip(df_e,0.05,0.95)
$$

再转为 IRT 难度参数：

$$
b_e=\log\frac{d_e}{1-d_e}
$$

#### 8.1.2 学生客观能力

学生在知识点 $k$ 上的历史能力记为：

$$
H_{u,k}
$$

对于习题 $e$，学生的题目级能力取该题相关知识点的平均值：

$$
H_{u,e}=\frac{1}{|K_e|}\sum_{k\in K_e}H_{u,k}
$$

同样进行裁剪后，转为 IRT 能力参数：

$$
\theta_{u,e}=\log\frac{H_{u,e}}{1-H_{u,e}}
$$

#### 8.1.3 客观成功概率

基于 IRT 思路，学生 $u$ 做对习题 $e$ 的客观成功概率为：

$$
P^{obj}_{u,e}=\sigma(\theta_{u,e}-b_e)
$$

其中 $\sigma(\cdot)$ 是 Sigmoid 函数。

#### 8.1.4 模型预测掌握概率

模型端使用 `MLKC` 表示学生对知识点的预测掌握程度。对于多知识点习题，使用几何平均得到题目级预测掌握概率：

$$
P^{model}_{u,e}=\left(\prod_{k\in K_e}MLKC_{u,k}\right)^{1/|K_e|}
$$

如果某个知识点缺少 `MLKC`，使用学生历史全局均值作为兜底。

#### 8.1.5 ACC@N 公式

单个学生的 `ACC@N` 定义为：

$$
ACC_u@N=\frac{1}{N}\sum_{e\in S_u^N}\left(1-\left|P^{obj}_{u,e}-P^{model}_{u,e}\right|\right)
$$

整体结果对所有测试学生求均值和标准差：

$$
ACC@N=Mean_u(ACC_u@N)
$$

该指标越高，说明模型对学生掌握状态的预测越接近客观估计。

### 8.2 ZPD-EMCC@N

`ZPD-EMCC@N` 用来衡量推荐列表的学习收益。它关注两个问题：

```text
1. 推荐题是否覆盖了学生后续真正需要学习的知识点。
2. 推荐题的难度是否适合学生当前能力水平。
```

对学生 $u$ 的前 $N$ 道推荐题，记为：

$$
S_u^N
$$

推荐列表覆盖到的知识点集合为：

$$
K(S_u^N)=\bigcup_{e\in S_u^N}K_e
$$

#### 8.2.1 题目适切度

题目归一化难度定义为：

$$
b^{norm}_e=1-df_e
$$

学生对习题 $e$ 的综合能力为：

$$
H_{u,e}=\frac{1}{|K_e|}\sum_{k\in K_e}H_{u,k}
$$

题目适切度定义为：

$$
Suit_{u,e}=1-|H_{u,e}-b^{norm}_e|
$$

当学生能力和题目难度接近时，$Suit_{u,e}$ 更高；当题目过难或过易时，$Suit_{u,e}$ 会降低。这个设计对应教育中的 ZPD 思想，即题目应尽量落在学生当前能力附近，而不是过于简单或过于困难。

#### 8.2.2 单题知识点习得率

对习题 $e$ 中的知识点 $k$，单题带来的习得贡献定义为：

$$
\alpha_{u,e,k}=\sigma\left(\gamma_s(Suit_{u,e}-H_{u,k})\right)
$$

其中 $\gamma_s$ 是缩放系数，当前代码中取：

$$
\gamma_s=2.0
$$

这个公式的含义是：当题目适切度高，且学生对该知识点尚未充分掌握时，推荐该题更可能带来有效学习收益。

#### 8.2.3 多题联合覆盖

如果推荐列表中多道题覆盖同一个知识点，收益不应简单线性累加。为体现边际收益递减，对知识点 $k$ 的联合覆盖函数定义为：

$$
f_{u,k}(S_u^N)=1-\exp\left(-\beta\sum_{e\in S_u^N, k\in K_e}\alpha_{u,e,k}\right)
$$

当前代码中：

$$
\beta=1.0
$$

当同一知识点被多道题覆盖时，$f_{u,k}$ 会逐渐接近 1，但不会无限增大。

#### 8.2.4 PKC 加权的学习收益

`PKC` 表示学生对知识点 $k$ 的后续学习需求。最终的原始 EMCC 分数为：

$$
EMCC^{raw}_u@N=\sum_{k\in K(S_u^N)}PKC_{u,k}\cdot f_{u,k}(S_u^N)
$$

该分数是加权求和，因此不限定在 $[0,1]$。如果推荐列表覆盖了多个高需求知识点，`EMCC(raw)` 可以大于 1。

为了便于观察推荐列表内部覆盖质量，也可以计算局部归一化版本：

$$
EMCC^{local}_u@N=\frac{EMCC^{raw}_u@N}{\sum_{k\in K(S_u^N)}PKC_{u,k}+\epsilon}
$$

其中分母只使用推荐列表实际覆盖到的知识点的 `PKC` 质量，而不是学生所有知识点的 `PKC` 总量。这个局部归一化版本更适合解释“当前推荐列表覆盖到的知识点质量如何”。

最终对所有学生求均值和标准差：

$$
ZPD\text{-}EMCC@N=Mean_u(EMCC_u@N)
$$

### 8.3 ACC 和 ZPD-EMCC 的区别

`ACC@N` 关注模型判断是否准确：

```text
模型预测的掌握概率是否接近客观 IRT 成功概率。
```

`ZPD-EMCC@N` 关注推荐是否有学习价值：

```text
推荐列表是否覆盖学生未来需要学习的知识点，并且题目难度是否适合当前学生。
```

两者并不完全等价。只推荐很简单的题，模型可能更容易判断学生能做对，`ACC` 可能较高，但学习收益不一定高。推荐更贴近学生最近发展区的题，`EMCC` 往往更能体现推荐列表的教育价值。

---

## 9. 代码结构

项目主要文件结构如下：

```text
KG4EX/
├── README.md
├── requirements.txt
├── codes/
│   ├── dataloader.py
│   ├── model.py
│   ├── run.py
│   └── test_models.py
├── data/
│   └── algebra2005/
│       ├── Q.txt
│       ├── entities.dict
│       ├── relations.dict
│       ├── generate_dict.py
│       ├── build_stats.py
│       └── raw_stats_akt.pkl
└── pyKT_example/
    └── output_final_akt_fpkc_h5_g09.csv
```

其中：

| 文件 | 作用 |
|---|---|
| `codes/model.py` | 定义 TransE、TripleRE、CogRE 等嵌入模型 |
| `codes/dataloader.py` | 负责正负样本构造和 DataLoader 数据组织 |
| `codes/run.py` | 训练入口，负责读取字典、三元组、训练模型并保存 checkpoint |
| `codes/test_models.py` | 统一测试入口，支持 CogRE、TripleRE 和 TransE-ADV |
| `data/algebra2005/generate_dict.py` | 根据 train/test triples 生成实体和关系字典 |
| `data/algebra2005/build_stats.py` | 根据 pyKT 输出 CSV 和 Q 矩阵生成离线评价统计 |
| `data/algebra2005/Q.txt` | 习题-知识点矩阵 |
| `data/algebra2005/raw_stats_akt.pkl` | ACC 和 EMCC 使用的离线题目正确率估计 |

---

## 10. 运行流程

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

### 10.2 生成字典

进入数据目录：

```bash
cd data/algebra2005
```

根据训练和测试三元组生成字典：

```bash
python generate_dict.py \
  --train_file akt_fpkc_h5_g09_triples.txt \
  --test_file akt_fpkc_h5_g09_test_triples.txt
```

该脚本会生成：

```text
entities.dict
relations.dict
```

### 10.3 生成离线评价统计

```bash
python build_stats.py \
  --csv_path ../../pyKT_example/output_final_akt_fpkc_h5_g09.csv \
  --q_file Q.txt
```

该脚本会生成：

```text
raw_stats_akt.pkl
```

其中包含：

```python
exercise_df_dict
global_df_mean
```

### 10.4 训练 CogRE

进入代码目录：

```bash
cd ../../codes
```

运行训练：

```bash
python run.py --do_train --cuda \
  --data_path ../data/algebra2005 \
  --save_path ./models/algebra2005/cog_h5_g09 \
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

如果希望训练更快并减少频繁保存，可以使用：

```bash
python -u run.py --do_train --cuda \
  --data_path ../data/algebra2005 \
  --save_path ./models/algebra2005/cog_h5_g09 \
  --model CogRE \
  -b 512 \
  -n 128 \
  -d 500 \
  -r 0.00 \
  -lr 0.001 \
  -cpu 20 \
  -3u 1.0 \
  --max_steps 30000 \
  -adv \
  --save_checkpoint_steps 5000 \
  --log_steps 500
```

### 10.5 测试 CogRE

```bash
python test_models.py \
  --model_type cogre \
  --embedding_path ./models/algebra2005/cog_h5_g09
```

### 10.6 测试 TripleRE

```bash
python test_models.py \
  --model_type triplere \
  --embedding_path ./models/algebra2005/triplere_h5_g09 \
  --triplere_u 1.0
```

如果训练 TripleRE 时使用的是其他 `u` 值，测试时必须保持一致。

### 10.7 测试 TransE-ADV

```bash
python test_models.py \
  --model_type transe_adv \
  --embedding_path ./models/algebra2005/transe_adv_h5_g09
```

---

## 11. 注意事项

1. `entities.dict` 和 `relations.dict` 必须和训练模型时使用的字典保持一致。模型 embedding 的行号依赖字典 ID，如果训练后重新生成字典，可能导致测试时读取错 embedding。

2. `test_models.py` 中的 CogRE 评分方向必须与 `model.py` 一致。当前方向为：

   $$
   diff=t'-(h'+r_m)
   $$

3. `raw_stats_akt.pkl` 应当由与当前实验匹配的 pyKT 输出 CSV 生成。如果训练和测试使用的是 `h5_g09`，则离线统计也应使用相同版本的数据源。

4. 大规模数据文件和训练好的 checkpoint 通常不建议直接上传到 GitHub。仓库中可以保留脚本、README、示例数据和必要的小型配置文件。

---

## 12. 当前阶段成果

目前，本项目已经完成了从学生特征提取、知识图谱构建、嵌入模型训练到推荐效果评价的完整实验流程。相比原始 KG4EER 流程，本项目主要在两个方面进行了改进。

首先，在学生特征提取阶段，项目引入了基于 Transformer 架构的序列建模方法，用于提取学生在历史学习过程中的知识掌握状态、后续知识点需求以及遗忘相关特征。相比传统序列模型，Transformer 能够更好地捕捉长距离学习行为之间的依赖关系，使学生状态表示更加充分。

其次，在知识图谱嵌入阶段，项目设计并使用了 CogRE 模型。该模型在关系建模时引入了关系门控和非对称距离度量，使不同类型的教育关系能够以更细粒度的方式影响实体表示。对于 `mlkc`、`pkc`、`exfr` 和 `rec` 这类具有明确教育语义的关系，CogRE 能够比普通平移模型更灵活地表示学生、知识点和习题之间的关联。

在评价方面，项目不再只依赖传统推荐指标，而是重点使用 `ACC@N` 和 `ZPD-EMCC@N` 两个与教育推荐目标更一致的指标。`ACC@N` 关注模型预测的学生掌握概率是否接近客观 IRT 估计结果；`ZPD-EMCC@N` 则关注推荐列表是否覆盖学生真正需要学习的知识点，并考虑题目难度与学生当前能力之间的适切性。因此，这两个指标不仅衡量推荐结果是否“准确”，也进一步衡量推荐内容是否具有潜在学习收益。

当前实验结果表明，改进后的流程在 `ACC` 和 `ZPD-EMCC` 两个核心指标上均表现出优于原始模型的趋势。这说明基于 Transformer 的学生特征提取、改进后的 CogRE 嵌入建模，以及面向教育收益的评价指标设计，能够更好地服务于个性化习题推荐任务。

需要说明的是，完整训练结果和多组实验对比仍在整理中，后续将进一步补充不同模型、不同参数设置下的详细数值结果。
