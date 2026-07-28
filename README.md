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
