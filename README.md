# Model Console

基于 FastAPI 的模型管理、聊天和评测控制台。支持连接 OpenAI 兼容接口、启动本机 vLLM、SSE 日志与聊天、并发评测、断点续跑、榜单及 Excel 导出。

## 快速开始

需要 Python 3.12+ 和 uv。进入项目根目录后执行:

```bash
uv sync --frozen
uv run --frozen python app.py
```

默认监听 `0.0.0.0:27000`。H200 现有目录是
`/home/users/caoshipeng/workspace/model_console`。通过本地 SSH 隧道访问:

```bash
ssh -N -L 27000:127.0.0.1:27000 H200
```

浏览器打开 `http://127.0.0.1:27000/`。控制台提供启动、停止模型和修改配置的管理能力,
当前没有登录鉴权,部署时应放在受信任网络、SSH 隧道或带鉴权的反向代理后。

仅连接已有模型接口时不需要 GPU 或 vLLM。需要从控制台启动本机模型时,
在与 CUDA、模型兼容的环境中单独安装 vLLM; 它没有包含在控制台的 uv 锁定依赖中。
程序依次查找当前 Python 环境、项目 `.venv`、父目录共享 `.venv` 和 `PATH` 中的 `vllm`。
模型相对权重路径仍按父目录 workspace 解析,绝对路径按原值使用。

## 目录与配置

| 路径 | 用途 |
| --- | --- |
| `app.py` | FastAPI API、模型生命周期、聊天和 SSE |
| `templates/index.html` | 单页前端,内联 CSS / JavaScript |
| `static/` | KaTeX、Lucide 和字体资源 |
| `model_config.py` | 内置模型定义、默认 vLLM 参数、配置合并 |
| `models_user.example.json` | H200 用户模型配置的脱敏示例 |
| `model_descriptions.json` | 模型说明 |
| `eval/` | 数据集注册、模型调用、评测、榜单及错题工具 |
| `tests/` | 离线回归测试 |
| `docs/REVIEW.md` | 本次整理范围、审核重点和验证结果 |
| `pyproject.toml` / `uv.lock` | Python 依赖声明和锁文件 |

`model_config.py` 位于项目根目录,是实体文件。每次请求通过 `_cfg()` 热重载;
模型配置编辑不需要重启服务。已有内置条目的权重路径和 GPU 设置属于原部署,
迁移时需按目标机器调整。

### 添加或覆盖模型

Web 界面的新增、编辑、删除操作写入 `models_user.json`。
同名 key 覆盖内置条目; 删除用户条目后会回退到同名内置定义。
此文件可能包含 API Key,已被 Git 忽略。

新部署可先参考 `models_user.example.json`,也可以复制后调整:

```bash
cp models_user.example.json models_user.json
```

示例保留当前 H200 注册模型的端口、接口和模型名,所有真实 API Key 已移除。
`REPLACE_WITH_YOUR_API_KEY` 必须替换为自己的密钥,远程地址、权重路径和 SSH 隧道也需按环境配置。
不要用模板覆盖已有部署的 `models_user.json`。

| API | 用途 |
| --- | --- |
| `GET /api/model-config/{key}` | 读取模型配置 |
| `POST /api/model-config` | 新增或覆盖,body 为 `{key, config}` |
| `DELETE /api/model-config/{key}` | 删除用户层配置 |

### 评测答案抽取模型

MCQ 判分和 Inverse IFEval judge 需要可用的答案抽取模型。
默认 key 为 `h100_deepseek_v41_flash_extract_temp`,其定义包含在用户配置示例中。
可通过环境变量指定自己的模型 key:

```bash
export MODEL_CONSOLE_EXTRACT_MODEL_KEY=your_extractor_key
uv run --frozen python app.py
```

也可以使用 `.env`:

```bash
cp .env.example .env
uv run --frozen --env-file .env python app.py
```

直接运行 `python app.py` 不会自动加载 `.env`。该环境变量为空时,
回退到 `http://127.0.0.1:26001/v1` 的 `Qwen/Qwen3.6-35B-A3B`。
API 密钥写在本地 `models_user.json` 对应条目的 `api_key` 中。

### 数据与运行产物

- `eval/data_general/`、`eval/data_sampled10/`、`eval/test_subset.jsonl` 是原仓库已跟踪的数据,本次保留。
- 完整医疗数据位于 H200 的 `eval/data_medical/` (约 14 MB),未纳入 Git。
  新部署需从原部署恢复该目录; 缺少时高难子集显示数据缺失、完整集不注册,可用列表以 `/api/eval/benches` 的 `exists` 字段为准。
- `eval/output/`、`eval/analysis/`、`eval/leaderboard.json`、日志、虚拟环境、NLTK 缓存与配置备份均不纳入 Git。
- `eval/medbench.json` 是人工维护的榜单快照,其中 `updated` 表示快照时间,不代表实时榜单。

## 开发验证

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m compileall -q app.py model_config.py eval tests
```

回归测试覆盖评测完整性判断、连接恢复、超时不重试、停止响应和 vLLM 环境查找,
不调用真实模型,不占用 GPU。

## Git 同步

GitHub 源仓库: `caobiabia/model-manager` (`master`)。
Codeup `SafeMedLLM` 的 `master` 分支将完整项目快照存放在
`code-csp/model-console/`,该目录与 GitHub 项目根目录对应。
Codeup 根目录已有文件保留。两处代码通过普通 Git 提交同步,不嵌套 `.git`、不使用子模块。

## 测评 (Eval)

测评功能移植自 `workspace/eval/`,将原来的 `benchmark.py`（MedicalAgentsBench 10 数据集）
和 `medqa_subset.py`（中文 MedQA 子集）统一为一套代码。

### 架构

| 文件 | 说明 |
| --- | --- |
| `eval/common.py` | 共享工具：路径、数据加载、客户端创建、模型调用、答案提取（统一 DeepSeek + 正则兜底） |
| `eval/benches.py` | Bench 注册表：按数据是否可用注册医疗与通用 bench |
| `eval/runner.py` | 统一 EvalRunner：model x bench 笛卡尔积，并发跑题，SSE 进度推送，断点续跑 |

### Bench 列表

10 个 MedicalAgentsBench 数据集的高难子集与完整测试集,以及 8 个通用 Bench。原有中文子集数据保留在仓库中,当前未注册。医疗数据需先按上文恢复。

`medqa` `pubmedqa` `medmcqa` `mmlu` `mmlu-pro` `medbullets` `afrimedqa` `medexqa` `medxpertqa-r` `medxpertqa-u` (完整集 key 加 `_full`)

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
