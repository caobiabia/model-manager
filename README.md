# model_console — vLLM 模型管理 & 聊天界面

一个基于 FastAPI 的 Web 应用,用于在本地启动/扫描 vLLM 模型并与之聊天。
后端在**前台进程**中拉起 vLLM,通过 SSE 实时推送日志;前端是单文件单页应用。

## H200 部署说明(2026-09-02 迁移)

本项目从 `yiling:/mntnlp/csp/workspace/csp_dev` 迁移到
`H200:/home/users/caoshipeng/workspace/model_console`,目录已更名
`csp_dev -> model_console`。本 README 其余部分描述的 yiling 旧路径
(`/mntnlp/...`)在 H200 上不存在,以下方替换关系为准:

| yiling 旧位置 | H200 新位置 |
| --- | --- |
| `/mntnlp/csp/workspace/csp_dev/` | `~/workspace/model_console/` |
| `/mntnlp/csp/workspace/model_config.py` | `model_console/model_config.py`(仓库内唯一一份,**当前为空占位**,需为 H200 填写模型条目;迁移时已去掉符号链接结构) |
| `/mntnlp/csp/workspace/.venv` | 尚未创建,见下方快速开始 |
| `serve.py` / `eval_models.py` 等上层脚本 | 暂未迁移,仍在 yiling |

医疗评测数据位置: MedicalAgentsBench 已收进仓库内
,另有
;两者均在  中(可从 yiling
 重导)。

H200 上**只有一份 `model_config.py`,就在仓库目录里**,直接编辑即可
(yiling 时代的父目录 + 符号链接结构已在迁移时拆除)。`app.py` 每次
请求热重载配置,改完即生效,不用重启服务。

### 快速开始(在 H200 上)

```bash
cd ~/workspace/model_console
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn httpx          # 界面/聊天所需最小依赖
# 需要用界面启动 vLLM 时再装: pip install vllm
python app.py                               # 监听 0.0.0.0:27000
```

浏览器访问 `http://<H200地址>:27000/`(公网被安全组挡住时可走 SSH 隧道:
`ssh -N -L 27000:127.0.0.1:27000 H200`)。

远端仓库仍为 `git@github.com:caobiabia/model-manager.git`,迁移时通过
`git clone` 保留了完整提交历史。

## 组成

| 文件 | 说明 |
| --- | --- |
| `app.py` | FastAPI 后端:启动 vLLM、端口扫描自动探测已在运行的模型、聊天转发、SSE 日志流、按需热重载配置 |
| `templates/index.html` | 单文件前端:内联 CSS/JS,使用 Lucide 图标、KaTeX 数学渲染、Inter 字体 |
| `model_config.py` | 模型配置唯一数据源,仓库内实体文件(H200 版已去符号链接) |

## model_config.py

`model_config.py` 是「启动、评测、benchmark 的唯一数据源」。
在 yiling 原部署中它放在上层 workspace 目录,由 `serve.py` 等兄弟脚本共享,仓库通过
`../model_config.py` 符号链接纳入管理;迁移到 H200 后这些脚本未随行,已简化为
**仓库内单份实体文件**。

`app.py` 启动时把所在目录加入 `sys.path` 后 `import model_config`,配置即随仓库
自洽;直接 `python -c "import model_config"` 也可用。

后端还提供 `_cfg()` —— 每次请求时 `importlib.reload(model_config)`,使配置编辑**无需重启服务**即可生效。

## 前端配置模型（无需手改 model_config.py）

平台使用者可以直接在 Web 界面「新增模型」或点击模型行上的齿轮图标「编辑/删除」模型，
无需手动编辑 `model_config.py`。前端配置会写入 **`models_user.json`**（与本仓库同目录），
`model_config.py` 加载时按 key 合并进 `MODELS`：

- **新增**：填入内置没有的 key，即追加一个模型；
- **覆盖**：key 与内置模型相同，用户的版本优先（例如只改 `display_name` 或 `gpu`）；
- **删除**：仅删除用户层配置；若对应 key 是内置模型，会回退为内置配置。

合并后的 `MODELS` 就是启动 / 评测 / benchmark 的唯一数据源，
下游脚本（`eval/common.py`、`eval/runner.py`、`eval/leaderboard.py` 等）通过
`from model_config import MODELS` 直接可见，无需改动。

后端 API：
- `GET  /api/model-config/{key}` —— 读取单模型完整配置（编辑表单回填用）；
- `POST /api/model-config` —— 新增或覆盖，body 为 `{key, config}`；
- `DELETE /api/model-config/{key}` —— 删除用户层配置。

> `models_user.json` 可能包含 API Key，已加入 `.gitignore`，不会进入版本管理。

## 运行环境

依赖(`fastapi` / `uvicorn` / `httpx`)由上层仓库的虚拟环境提供:

```
/mntnlp/csp/workspace/.venv
```

`app.py` 启动 vLLM 时也使用该 `.venv/bin/vllm` 与 `.venv/bin/python`。

## 启动

```bash
cd /mntnlp/csp/workspace/csp_dev
python app.py
```

默认监听 `0.0.0.0:27000`,`reload=True`(监控 `csp_dev/` 内的 app/模板改动)。
浏览器访问 `http://<host>:27000/`。

## 配置(环境变量)

从环境变量读取,无硬编码密钥:

- `DEEPSEEK_API_KEY` —— 当模型 `provider=deepseek`(远程 API)时需要。
  复制 `.env.example` 为 `.env` 填入真实值,或 `export DEEPSEEK_API_KEY=...`。

## 注意

- 本仓库是 `/mntnlp/csp/workspace/csp_dev` 的**独立 git 仓库**,与上层 vLLM 上游仓库无共享历史。
- `__pycache__/`、`.venv/`、`.env`、`*.log` 等见 `.gitignore`,不会进入版本管理。

## 测评 (Eval)

测评功能移植自 `workspace/eval/`,将原来的 `benchmark.py`（MedicalAgentsBench 10 数据集）
和 `medqa_subset.py`（中文 MedQA 子集）统一为一套代码。

### 架构

| 文件 | 说明 |
| --- | --- |
| `eval/common.py` | 共享工具：路径、数据加载、客户端创建、模型调用、答案提取（统一 DeepSeek + 正则兜底） |
| `eval/benches.py` | Bench 注册表：11 个 bench 平铺注册，无 benchmark/subset 之分 |
| `eval/runner.py` | 统一 EvalRunner：model x bench 笛卡尔积，并发跑题，SSE 进度推送，断点续跑 |

### Bench 列表

10 个 MedicalAgentsBench 数据集（`test_hard` split）+ 1 个自定义中文子集 + 5 个通用 Bench：

`medqa` `pubmedqa` `medmcqa` `mmlu` `mmlu-pro` `medbullets` `afrimedqa` `medexqa` `medxpertqa-r` `medxpertqa-u` `medqa_cn`

通用 Bench（`eval/data_general/`），UI 中按能力分三组展示：「世界知识」(`mmlu_std`/`gsm8k`/`ceval`)、
「推理能力」(`gpqa_diamond`/`aime2026`)、「指令遵循」(`ifeval`/`ifbench`/`inverse_ifeval`)：

`mmlu_std`（标准 MMLU，57 科 14,042 题） `gsm8k`（GSM8K test，1,319 题，自由数字作答） `ceval`（C-Eval test，52 科 12,342 题） `gpqa_diamond`（GPQA Diamond，198 题） `aime2026`（AIME 2026，30 题，整数作答）

**指令遵循 Bench（`format="follow"`，由 verifier/judge 判分）**：

`ifeval`（Google IFEval，~540 prompt，25 类可验证指令，strict prompt-level） `ifbench`（Allen AI IFBench，300 样本、58 种 WildChat 约束） `inverse_ifeval`（Inverse IFEval，1,012 题、中英双语、8 类逆向任务，LLM-as-a-Judge）

数据由 `eval/prepare_general_benches.py` 从 HuggingFace 下载并转换为统一 JSONL 格式
（默认走 hf-mirror.com，可用 `HF_ENDPOINT` 覆盖）。GSM8K 是自由作答 bench
（`format="free"`），runner 用正则提取末尾数字并与标准答案精确匹配。GPQA / AIME 2026
由 `eval/prepare_extra_benches.py` 下载转换（GPQA 原仓库在 HuggingFace 上需申请访问权限，
脚本使用公开镜像的同一 Diamond 子集）。

三个指令遵循 Bench 由 `eval/prepare_follow_benches.py` 下载转换（源：`google/IFEval`、
`allenai/IFBench_test`、`m-a-p/Inverse_IFEval`）。判分方式：
- `ifeval` / `ifbench` 复用 `eval/follow/` 内移植的官方启发式 verifier（Apache-2.0），
  采用 **strict** prompt-level 指标（每条可验证指令都必须满足）。
- `inverse_ifeval` 使用数据自带的 judge prompt 走 **LLM-as-a-Judge**（复用本地
  DeepSeek 判卷模型，0/1 分）。

> 新增 Python 依赖：`nltk` / `emoji` / `syllapy` / `langdetect` / `absl-py` /
> `immutabledict`。NLTK 语料（`punkt` / `punkt_tab` / `stopwords` /
> `averaged_perceptron_tagger_eng`）在首次运行 verifier 时自动下载到
> `eval/follow/.nltk_data`（已 gitignore）。

### 扩展

在 `eval/benches.py` 末尾调用 `register(Bench(...))` 即可添加新 bench。只需提供 `id`、`name`、`data_file`、`language`，runner 自动处理其余逻辑。
新增自由作答类 bench（如数学题）时设置 `format="free"`，并把标准答案写入记录的 `answer` 字段；
新增指令遵循类 bench 时设置 `format="follow"` 并传入 `scorer(raw_response, question) -> bool`
（内置 verifier：IFEval/IFBench 用 `eval.follow.verify_follow_strict`，Inverse IFEval 用
`eval.follow.judge.judge_inverse_ifeval`）。

### 使用

浏览器打开后切换到「测评」标签，选择模型和 bench，点击开始即可。进度实时推送，结果自动保存到 `eval/output/<run_id>/`。

### 错题提取与分析

`eval/wrong_answers.py` 从已完成的 run 里导出逐题错题并出统计，不重新调用模型：

```bash
# 单模型：导出全部错题 JSONL + 分数据集统计 + REPORT.md
python eval/wrong_answers.py extract --run-id 20260910_003815 \
    --model-key qwen36_35b_sft_med_v72_ep3 --out eval/analysis/v72ep3

# 两模型对比：共错/单边错题、Jaccard、逐 bench 交叉表
python eval/wrong_answers.py compare --a eval/analysis/v72ep3 \
    --b eval/analysis/qwen38flash --out eval/analysis/compare
```

默认覆盖 MedicalAgentsBench 10 套数据的高难子集（`<ds>`）与全量测试集（`<ds>_full`），
可用 `--benches` 指定其它 bench。输出含题目/选项/gold/模型答案/完整 `raw_response`/reasoning，
以及错误类型（未作答 vs 误选）、选项分布、生成长度等统计。`compare` 额外产出
`right_b_wrong_a.jsonl` / `right_a_wrong_b.jsonl`——单边错题的富记录（题干、选项、gold、
双方答案字母与完整回答），可直接拿去做差距分析。`eval/analysis/` 为分析产物目录（已 gitignore）。

### 榜单

「榜单」标签顶部是一排分榜名字，点哪个看哪个（不再把全部分榜竖着堆在一页）：

| 分榜 | 来源 |
| --- | --- |
| MedBench | 官方榜单快照，人工维护在 `eval/medbench.json` |
| MedicalAgentsBench 高难子集 (test_hard) / 全量测试集 (test) | 内部测评导入的成绩：10 套 MedicalAgentsBench 数据集的高难子集与完整测试集 |
| 指令遵循 / 世界知识 / 推理能力 | 内部测评导入的通用 Bench 成绩 |

MedBench 分榜含两张表：**API 榜单**（官方 API 评测）与**自测榜单**（提交到 MedBench
自测），两张表列结构相同——排名 / 模型 / 综合得分 + 医学知识问答 / 医学语言生成 /
复杂医学推理 / 医学语言理解 / 医疗安全和伦理，均按综合得分降序排名。这些分数
**不来自本仓库的测评流程**，而是官方站点的快照，直接编辑 `eval/medbench.json`
（`api` / `self_test` / `sub_benches` 三段）即可更新，保存后刷新页面生效，无需重启服务；
`self_test` 里官网附带的提交日期、组织、参数量等字段仍保留在 JSON 中，只是不在表里展示。
表里的模型名统一用内部 checkpoint 正式名（即 `model_config.py` 的 `display_name`），
例如 MedBench 站上的 `Med-v7.2` 在本仓库写作
`Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.2-epoch3)`，这样同一模型在不同分榜里
读到的名字一致；非本团队的模型（百度灵医智惠、qiaojian-med 等）按官方叫法保留。

内部榜单条目来自「导入历史记录」（把已完成 run 的成绩写进 `eval/leaderboard.json`），
用顶部「长度」切换 256k / 16k 两个上下文长度榜；MedBench 分榜不参与该筛选。
已下架的历史分榜（`general_subset` 通用子集、`s10` 采样 10%、`medqa_cn` 自定义子集）
不再出现在榜单里，但条目仍保留在 `eval/leaderboard.json`，需要时把分类加回
`eval/leaderboard.py` 的 `get_leaderboard()` 即可恢复。
