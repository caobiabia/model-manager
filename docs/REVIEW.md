# Model Console 代码审核说明

整理日期: 2026-10-21 (Asia/Shanghai)

## 来源与范围

- 源目录: `H200:/home/users/caoshipeng/workspace/model_console`。
- GitHub: `caobiabia/model-manager`, 分支 `master`。
- 源基线: `1ebe08f3be3a6a5fe2a507c0a727d8fa103fedb2`。
- 采集源仓库全部已跟踪文件,纳入 H200 当时的 7 个未提交文件改动。
- Codeup: `SafeMedLLM`, `master` 分支的 `code-csp/model-console/`。
- GitHub 项目根目录与 Codeup 上述子目录内容相同; Codeup 根目录的初始化文件保留。

整理在独立副本完成。没有修改 H200 的代码、真实用户配置、评测任务或运行进程。
本次同步代码,没有迁移权重、虚拟环境、评测输出或生产数据库。

## 纳入的 H200 改动

| 文件 | 审核重点 |
| --- | --- |
| `app.py` | 开始/恢复评测前探测真实模型接口; 使用 runner 的完成状态; 读取运行中状态时优先内存记录 |
| `eval/common.py` | 连接故障可等待恢复; 恢复窗口从故障发生时开始; 等待期间响应停止; 生成超时不重试 |
| `eval/runner.py` | 缺题、未覆盖的模型/数据集组合、基础设施错误不标成完成; 保留失败记录; 支持已有配置中的自动重试; 进度计数使用已保存结果 |
| `templates/index.html` | 前端保留 interrupted 状态; SSE 结束后核对服务端状态 |
| `model_config.py` | 同步当前 H200 使用的答案抽取模型 key |
| `model_descriptions.json` | 同步 med-v8.1、MAB2 多个 checkpoint 的部署说明 |
| `eval/medbench.json` | 同步 H200 已有的人工榜单快照,不是重新联网核验的实时成绩 |

## 本次整理补充

1. 修正 vLLM 程序固定指向父目录虚拟环境的问题。支持当前 Python 环境、项目环境、旧共享环境和 `PATH`,保留旧部署兼容性。
2. 明确从项目目录导入 `model_config.py`,修正“空占位配置”和旧 yiling 部署文案。
3. 增加 `MODEL_CONSOLE_EXTRACT_MODEL_KEY` 环境变量,默认仍为 H200 现有 key。
4. 增加 `models_user.example.json`,来自当前 10 条用户模型配置,API Key 替换为占位符; URL 用户信息和查询参数移除。
5. 更新 README 和 `.env.example`,给出锁定依赖安装、SSH 访问、数据恢复和验证命令。明确 `.env` 需用 `uv run --env-file` 显式加载。
6. 增加忽略规则,排除 `.env` 变体、用户配置备份、配置备份目录、临时文件和 `d2.txt`。
7. 增加不调用模型、不占用 GPU 的离线回归测试。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| `uv sync --frozen` | 通过,使用现有 `uv.lock` 创建 Python 3.12 环境 |
| `python -m unittest discover -s tests -v` | 11 项在本地 macOS 和 H200 Linux 临时目录均通过 |
| Python 源码编译 | `app.py`、`model_config.py`、`eval/`、`tests/` 在两端均通过 |
| 前端内联 JavaScript | `node --check` 通过 |
| 项目 JSON | 全部解析通过 |
| 已跟踪 JSONL 数据 | 49 个文件、34,002 条记录全部解析通过 |
| ASGI 冒烟检查 | 首页、KaTeX、Lucide、数据集列表、评测历史均返回 200; 空评测请求返回 400 |
| 凭据检查 | 当前交付文件未发现私钥、常见 token 或非占位的固定密钥匹配 |

11 项测试覆盖: 完整/缺失评测结果、基础设施失败、主动限题、已有结束状态、
连接恢复与故障时间预算、生成超时不重试、停止中断等待、项目/旧共享/当前 vLLM 环境。

未进行真实模型推理、GPU 启动或完整基准评测。凭据检查是静态规则扫描,不等于全历史安全审计。

## 需要审核的现有约束

- **多任务停止隔离**: `EvalRunner.stop()` 调用进程级 `request_abort()`,
  bench scope 也仅由 `model_key|bench_id` 组成。多任务同时运行时,停止一个任务可能影响其他任务。
  这是现有行为,本次保留; 后续应按 run_id 隔离停止范围与流连接。
- **管理端访问边界**: 当前 API 没有登录鉴权,可以操作模型和读取配置。
  现阶段按内部管理工具使用,通过受信任网络、SSH 或带鉴权代理访问。
- **部署数据不在代码中**: `models_user.json`、`eval/data_medical/` (约 14 MB)、
  `eval/leaderboard.json` 和 `eval/output/` 保留在 H200。新部署需恢复配置及数据。
  没有医疗数据的代码副本注册 18 个 bench,其中 8 个通用 bench 有数据,
  10 个医疗高难 bench 的 `exists=false`,医疗完整集暂不注册。
- **示例不是生产配置**: 示例保留 H200 的模型名、端口和机器路径,需要有效接口、
  隧道及自己的 API Key 才能调用。当前默认抽取模型的定义位于被忽略的用户配置层。
- **自动重试是显式选择**: runner 的 `auto_retry_incomplete` 默认关闭,
  恢复已有任务时读取配置值。开启后按 30 秒间隔继续重试,直到恢复或手动停止。

## 审核建议顺序

先看 `eval/runner.py` 的 `summary_status()` 和 `eval/common.py` 的连接恢复策略,
再看 `app.py` 与前端状态对齐。迁移部署时重点核对 README、模型示例和上述现有约束。
