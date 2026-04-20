from __future__ import annotations
from dataclasses import dataclass
from typing import Final

@dataclass
class Config:
    model_repo: str
    served_model_name: str
    model_file: str | None

    gpu_type: str
    n_gpu: int
    max_model_len: int
    max_num_seqs: int

    engine: str
    port: int


class Globals:
    HF_CACHE_ROOT_PATH: Final[str] = f"/root/.cache/huggingface"
    PREWARM_TIMEOUT_S: Final[int] = 60 * 60  # how long should we wait for the prewarm function to complete?
    TIMEOUT_S: Final[int] = 30 * 60  # how long should we wait for container start?
    SCALEDOWN_S: Final[int] = 8 * 60  # how long should we stay up with no requests?

    ALLOWED_ROUTES: Final[dict[str, set[str]]] = {
        "/v1/models": {"GET", "HEAD", "OPTIONS"},
        "/v1/chat/completions": {"POST", "OPTIONS"},
    }

    # For now, hardcode it. Later on, read it from an .env file
    API_KEYS: Final[list[str] | None] = None

    
def get_active_config() -> Config:
    return QWEN3_5_122B_SEHYO_NVFP4


GEMMA4_31B_BF16 = Config(
    model_repo="google/gemma-4-31B-it",
    served_model_name="google/gemma-4-31B-it-BF16",
    model_file=None,
    gpu_type="B200+",
    n_gpu=1,
    max_model_len=128000,
    max_num_seqs=8,
    engine="vllm",
    port=8000,
)

GEMMA4_31B_NVFP4 = Config(
    model_repo="nvidia/Gemma-4-31B-IT-NVFP4",
    served_model_name="nvidia/Gemma-4-31B-IT-NVFP4",
    model_file=None,
    gpu_type="B200+",
    n_gpu=1,
    max_model_len=128000,
    max_num_seqs=8,
    engine="vllm",
    port=8000,
)

QWEN3_5_122B_FP8 = Config(
    model_repo="Qwen/Qwen3.5-122B-A10B-FP8",
    served_model_name="Qwen/Qwen3.5-122B-A10B-FP8",
    model_file=None,
    gpu_type="B200+",
    n_gpu=1,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)

QWEN3_5_122B_AWS = Config(
    model_repo="QuantTrio/Qwen3.5-122B-A10B-AWQ",
    served_model_name="QuantTrio/Qwen3.5-122B-A10B-AWQ",
    model_file=None,
    gpu_type="B200+",
    n_gpu=1,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)

QWEN3_5_122B_INT4 = Config(
    model_repo="Intel/Qwen3.5-122B-A10B-int4-AutoRound",
    served_model_name="Intel/Qwen3.5-122B-A10B-int4-AutoRound",
    model_file=None,
    gpu_type="B200+",
    n_gpu=1,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)


QWEN3_5_122B_SEHYO_NVFP4 = Config(
    model_repo="Sehyo/Qwen3.5-122B-A10B-NVFP4",
    served_model_name="Sehyo/Qwen3.5-122B-A10B-NVFP4",
    model_file=None,
    gpu_type="B200+",
    #gpu_type="H200",
    #gpu_type="A100-80GB",
    n_gpu=1,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)


QWEN3_5_397B_NVFP4 = Config(
    model_repo="nvidia/Qwen3.5-397B-A17B-NVFP4",
    served_model_name="nvidia/Qwen3.5-397B-A17B-NVFP4",
    model_file=None,
    gpu_type="B200+",
    n_gpu=2,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)

QWEN3_5_397B_GPTQ = Config(
    model_repo="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    served_model_name="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
    model_file=None,
    gpu_type="B200+",
    n_gpu=2,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)

QWEN3_5_397B_AWQ = Config(
    model_repo="QuantTrio/Qwen3.5-397B-A17B-AWQ",
    served_model_name="QuantTrio/Qwen3.5-397B-A17B-AWQ",
    model_file=None,
    gpu_type="B200+",
    n_gpu=2,
    max_model_len=32000,
    max_num_seqs=64,
    engine="vllm",
    port=8000,
)

