"""
模型配置 — 启动、评测、benchmark 的唯一数据源。

!!! H200 占位版本 (2026-09-02 迁移自 yiling:/mntnlp/csp/workspace/model_config.py) !!!
原文件里的 11 个模型条目依赖 yiling 的 /mntnlp 路径与 GPU 拓扑,在 H200 上
不适用,因此这里 MODELS 留空。文件保留了与原版完全一致的加载/合并/辅助函数,
model_console 的 app.py 无需修改即可 import 并启动(界面上暂时一个模型都没有)。

用法: 在下方 MODELS 字典里按 key 添加 H200 上的模型,然后:
  启动 model_console:  cd ~/workspace/model_console && python app.py   # 监听 :27000
  (serve.py / eval_models.py 等 yiling 上层脚本暂未迁移,需要时从
   yiling:/mntnlp/csp/workspace/ 同步过来。)

字段说明 (与 yiling 原版一致):
  display_name      展示名称 (日志/汇总用)
  served_model_name vLLM --served-model-name / API model id
  model_path        本地模型权重路径 (vLLM serve 用)
  port              vLLM 端口
  gpu               CUDA_VISIBLE_DEVICES
  vllm_args         vLLM CLI 参数 (覆盖 DEFAULT_VLLM_ARGS)
  provider          "vllm"(默认) 或 "deepseek" (远程 API, 无需 model_path/port/gpu)
  base_url          仅 provider=deepseek 时指定
  base_model_path   可选 — 基模路径 (与 lora_path 配合, 替代 model_path)
  lora_path         可选 — LoRA 适配器目录 (与 base_model_path 配合)
  max_lora_rank     可选 — LoRA 最大秩, 默认 64 (仅 lora 模式)
  api_key           仅 provider=deepseek 时指定
  thinking          可选 — 思考强度(reasoning effort)能力声明。缺省 = 该模型不支持强度调节
                    (前端不展示控件, 只有思考开/关)。各模型值域不统一, 故不做强行归一化,
                    由模型自声明、前端按声明渲染:
                      transport  "chat_template_kwargs"(默认, 塞进 chat_template_kwargs)
                                 或 "top_level"(OpenAI 原生顶层 reasoning_effort 字段)
                      efforts    有序 dict {档位label: 实际下发值}, 值可为字符串或数字
                                 如 {"low": 25, "high": 50} → 前端下拉显示 "low (25)"
                      numeric    可选 {"min": .., "max": ..} — 额外允许自定义原始数值
                      default    可选 — 数值/档位对应的前端默认选中值

示例:
  MODELS["qwen38_flash_next"] = {
      "display_name": "Qwen3.8-Flash-Next",
      "served_model_name": "Qwen3.8-Flash-Next",
      "model_path": "/home/users/caoshipeng/models/Qwen3.8-Flash-Next",
      "port": 26011,
      "gpu": "0",
      "vllm_args": {"max-model-len": 262144},
  }
"""

import json
from pathlib import Path

# vLLM 通用默认参数 (所有模型共享, 可在单个模型 vllm_args 中覆盖)
DEFAULT_VLLM_ARGS = {
    "host": "0.0.0.0",
    "dtype": "bfloat16",
    "trust-remote-code": True,
    "async-scheduling": True,
    "max-model-len": 262144,
    # 让 vLLM 在输出侧就把思考内容与正文分离 (response 的 reasoning 字段);
    # 前端/评测侧仍保留从 content 剥离思考文本的兜底解析
    "reasoning-parser": "qwen3",
}

MODELS: dict = {
    "qwen36_35b_sft_med_ep1": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medical-mix-v7.0-epoch1)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.0-epoch1",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench/v7-20260907-125333/checkpoint-60",
        "port": 26021,  # 原 26012,让位给 DeepSeek-V4.1-Flash
        "gpu": 0,
        "vllm_args": {"max-model-len": 262144},
    },
    "qwen36_35b_sft_med_ep2": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medical-mix-v7.0-epoch2)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.0-epoch2",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench/v7-20260907-125333/checkpoint-120",
        "port": 26013,
        "gpu": 1,
        "vllm_args": {"max-model-len": 262144},
    },
    "qwen36_35b_sft_med_ep3": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medical-mix-v7.0-epoch3)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.0-epoch3",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench/v7-20260907-125333/checkpoint-180",
        "port": 26014,
        "gpu": 2,
        "vllm_args": {"max-model-len": 262144},
    },
    "qwen36_35b_sft_med_v71_ep1": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.1-epoch1)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.1-epoch1",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.1/v0-20260908-232807/checkpoint-151",
        "port": 26015,
        "gpu": 4,
    },
    "qwen36_35b_sft_med_v71_ep2": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.1-epoch2)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.1-epoch2",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.1/v0-20260908-232807/checkpoint-302",
        "port": 26016,
        "gpu": 5,
    },
    "qwen36_35b_sft_med_v71_ep3": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.1-epoch3)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medical-mix-v7.1-epoch3",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.1/v0-20260908-232807/checkpoint-453",
        "port": 26017,
        "gpu": 6,
    },
    "qwen36_35b_sft_med_v72_ep1": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.2-epoch1)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medbench-instruct-v7.2-epoch1",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-224",
        "port": 26018,
        "gpu": 7,
    },
    "qwen36_35b_sft_med_v72_ep2": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.2-epoch2)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medbench-instruct-v7.2-epoch2",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-448",
        "port": 26019,
        "gpu": 0,
    },
    "qwen36_35b_sft_med_v72_ep3": {
        "display_name": "Qwen3.6-35B-A3B (full-sft-medbench-instruct-v7.2-epoch3)",
        "served_model_name": "Qwen3.6-35B-A3B-full-sft-medbench-instruct-v7.2-epoch3",
        "model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-672",
        "port": 26020,
        "gpu": 1,
    },
    # ── v7.3 LoRA (基模 = v7.2 full-sft epoch3 / checkpoint-672) ──
    "qwen36_35b_lora_v73_ep8": {
        "display_name": "Qwen3.6-35B-A3B (lora-sft-medbench-instruct-v7.3-epoch8)",
        "served_model_name": "Qwen3.6-35B-A3B-lora-sft-medbench-instruct-v7.3-epoch8",
        "base_model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-672",
        "lora_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-lora-sft-medagentbench-medbench-instruct-v7.3/v1-20260914-174017/checkpoint-80",
        "max_lora_rank": 64,
        "port": 26022,
        "gpu": 0,
        "vllm_args": {"max-model-len": 262144},
    },
    "qwen36_35b_lora_v73_ep9": {
        "display_name": "Qwen3.6-35B-A3B (lora-sft-medbench-instruct-v7.3-epoch9)",
        "served_model_name": "Qwen3.6-35B-A3B-lora-sft-medbench-instruct-v7.3-epoch9",
        "base_model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-672",
        "lora_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-lora-sft-medagentbench-medbench-instruct-v7.3/v1-20260914-174017/checkpoint-90",
        "max_lora_rank": 64,
        "port": 26023,
        "gpu": 1,
        "vllm_args": {"max-model-len": 262144},
    },
    "qwen36_35b_lora_v73_ep10": {
        "display_name": "Qwen3.6-35B-A3B (lora-sft-medbench-instruct-v7.3-epoch10)",
        "served_model_name": "Qwen3.6-35B-A3B-lora-sft-medbench-instruct-v7.3-epoch10",
        "base_model_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-full-sft-medagentbench-medbench-instruct-v7.2/v0-20260909-172432/checkpoint-672",
        "lora_path": "/data1/users/caoshipeng/outputs/Qwen3.6-35B-A3B-lora-sft-medagentbench-medbench-instruct-v7.3/v1-20260914-174017/checkpoint-100",
        "max_lora_rank": 64,
        "port": 26024,
        "gpu": 2,
        "vllm_args": {"max-model-len": 262144},
    },
    "deepseek_v41_flash": {
        "display_name": "DeepSeek-V4.1-Flash",
        "served_model_name": "DeepSeek-V4.1-Flash",
        "model_path": "/data1/share-folder/models/DeepSeek-V4.1-Flash",
        "port": 26012,
        "gpu": "4,5,6,7",
        # 思考强度: chat_template_kwargs.reasoning_effort ∈ low(25)/high(50)/xhigh(75)/max(100)
        # 或原始整数 1-100; 不传时模型默认 ON @ 50 (minimal/medium 不是合法档位)
        "thinking": {
            "transport": "chat_template_kwargs",
            "efforts": {"low": 25, "high": 50, "xhigh": 75, "max": 100},
            "numeric": {"min": 1, "max": 100},
            "default": 50,
        },
        "vllm_args": {
            "max-model-len": 1048576,
            # 覆盖 DEFAULT_VLLM_ARGS 的 qwen3 解析器
            "reasoning-parser": "deepseek_v41",
            "tokenizer-mode": "deepseek_v41",
            "tool-call-parser": "deepseek_v41",
            "enable-auto-tool-choice": True,
            "tensor-parallel-size": 4,
            "speculative-config": '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}',
            "mm-encoder-tp-mode": "data",
            "gpu-memory-utilization": 0.97,
            "max-num-seqs": 256,
            "max-cudagraph-capture-size": 256,
        },
    },
    "qwen36_35b_extract": {
        "display_name": "Qwen3.6-35B-A3B",
        "served_model_name": "Qwen/Qwen3.6-35B-A3B",
        "model_path": "/home/users/caoshipeng/workspace/models/Qwen3.6-35B-A3B",
        "port": 26001,
        "gpu": 3,
        "vllm_args": {"max-model-len": 262144},
    },
}

# ──────────── 用户级覆盖 / 新增 (前端配置写入) ────────────
#
# 除上方内置 MODELS 外, 平台使用者可通过 Web 前端添加/修改/删除模型,
# 这些条目持久化在 models_user.json 中, 加载时按 key 合并进 MODELS。
#   覆盖: 与内置 key 相同 → 用户的版本优先
#   新增: 内置不存在的 key → 追加
#   删除: 从 models_user.json 移除该 key → 回退到内置(若有)

_USER_MODELS_FILE = Path(__file__).resolve().parent / "models_user.json"


def load_user_models() -> dict:
    """读取用户级模型配置 (mock 安全: 任何解析失败都返回空)。"""
    try:
        if _USER_MODELS_FILE.exists():
            data = json.loads(_USER_MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(k): v
                    for k, v in data.items()
                    if isinstance(v, dict)
                }
    except Exception:
        pass
    return {}


MODELS = {**MODELS, **load_user_models()}

# ──────────── 辅助函数 ────────────

def get_base_url(cfg: dict) -> str:
    """从 port 推导 base_url (vllm 模型) 或直接返回 base_url (deepseek)。"""
    if "base_url" in cfg:
        return cfg["base_url"]
    return f"http://localhost:{cfg['port']}/v1"


def get_eval_models() -> dict:
    """返回 eval_models.py 格式的模型字典。"""
    result = {}
    for key, cfg in MODELS.items():
        if cfg.get("provider") == "deepseek":
            continue
        result[key] = {
            "name": cfg["display_name"],
            "base_url": get_base_url(cfg),
            "model": cfg["served_model_name"],
        }
    return result


def get_local_models() -> dict:
    """返回 run_benchmark.py 格式的模型字典 (以 served_model_name 为 key)。"""
    result = {}
    for cfg in MODELS.values():
        if cfg.get("provider") == "deepseek":
            continue
        result[cfg["served_model_name"]] = {
            "base_url": get_base_url(cfg),
            "api_key": "not-needed",
        }
    return result


def get_model_by_name(name: str) -> dict | None:
    """通过 config key 或 served_model_name 查找模型配置。"""
    if name in MODELS:
        return MODELS[name]
    for cfg in MODELS.values():
        if cfg.get("served_model_name") == name:
            return cfg
    return None
