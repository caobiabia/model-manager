# csp_dev — vLLM 模型管理 & 聊天界面

一个基于 FastAPI 的 Web 应用,用于在本地启动/扫描 vLLM 模型并与之聊天。
后端在**前台进程**中拉起 vLLM,通过 SSE 实时推送日志;前端是单文件单页应用。

## 组成

| 文件 | 说明 |
| --- | --- |
| `app.py` | FastAPI 后端:启动 vLLM、端口扫描自动探测已在运行的模型、聊天转发、SSE 日志流、按需热重载配置 |
| `templates/index.html` | 单文件前端:内联 CSS/JS,使用 Lucide 图标、KaTeX 数学渲染、Inter 字体 |
| `model_config.py` | 符号链接 → `../model_config.py`(见下文) |

## model_config.py

`model_config.py` 是「启动、评测、benchmark 的唯一数据源」,定义了全部模型(11 个)。
它位于**父目录** `/mntnlp/csp/workspace/model_config.py`,不在本仓库目录内,
因此本仓库通过一个相对符号链接 `model_config.py -> ../model_config.py` 把它纳入版本管理。

`app.py` 用 `_workspace = Path(__file__).resolve().parents[1]` 把父目录加入 `sys.path`,
再 `import model_config`;运行时无需依赖符号链接即可正常导入。符号链接主要用于:
仓库可追踪该配置文件、以及在 `csp_dev/` 目录直接 `python -c "import model_config"` 时可用。

后端还提供 `_cfg()` —— 每次请求时 `importlib.reload(model_config)`,使配置编辑**无需重启服务**即可生效。

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

通用 Bench（`eval/data_general/`，UI 中归入「通用 Bench」分组）：

`mmlu_std`（标准 MMLU，57 科 14,042 题） `gsm8k`（GSM8K test，1,319 题，自由数字作答） `ceval`（C-Eval test，52 科 12,342 题） `gpqa_diamond`（GPQA Diamond，198 题） `aime2026`（AIME 2026，30 题，整数作答）

数据由 `eval/prepare_general_benches.py` 从 HuggingFace 下载并转换为统一 JSONL 格式
（默认走 hf-mirror.com，可用 `HF_ENDPOINT` 覆盖）。GSM8K 是自由作答 bench
（`format="free"`），runner 用正则提取末尾数字并与标准答案精确匹配。GPQA / AIME 2026 /
由 `eval/prepare_extra_benches.py` 下载转换（GPQA 原仓库在 HuggingFace 上需申请访问权限，
脚本使用公开镜像的同一 Diamond 子集）。

> SWE-bench Verified 暂缓接入：本环境没有 Docker，无法跑官方 harness 出分。`eval/benches.py`
> 里保留了注册代码（已注释），等有 Docker 评测环境后取消注释、运行
> `eval/prepare_extra_benches.py` 即可恢复；`eval/export_swebench_predictions.py` 用于把
> 生成的 patch 导出成官方 harness 的 predictions 格式。

### 扩展

在 `eval/benches.py` 末尾调用 `register(Bench(...))` 即可添加新 bench。只需提供 `id`、`name`、`data_file`、`language`，runner 自动处理其余逻辑。
新增自由作答类 bench（如数学题）时设置 `format="free"`，并把标准答案写入记录的 `answer` 字段；
新增补丁生成类 bench（如 SWE-bench）时设置 `format="patch"` 与 `scorable=False`。

### 使用

浏览器打开后切换到「测评」标签，选择模型和 bench，点击开始即可。进度实时推送，结果自动保存到 `eval/output/<run_id>/`。
