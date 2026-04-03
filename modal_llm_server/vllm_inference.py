# pyright: reportUnknownMemberType=false

"""
Deploy via: modal deploy vllm_inference.py
"""

import modal

MODEL_ID = "Qwen/Qwen3.5-27B"
SERVED_MODEL_NAME = f"{MODEL_ID}-BF16"
MODEL_REVISION = "b7ca741b86de18df552fd2cc952861e04621a4bd"  # 2026/02/25



IMAGE="vllm/vllm-openai:latest-cu130"
# IMAGE="vllm/vllm-openai:qwen3_5-cu130"
# IMAGE="vllm/vllm-openai:qwen3_5"

GPU_TYPE="B200+"
# GPU_TYPE="H200"
# GPU_TYPE="A100-80GB"
N_GPU=1
MAX_MODEL_LEN=16000
MAX_NUM_SEQS=90


TIMEOUT_S = 10 * 60  # how long should we wait for container start?
SCALEDOWN_S = 5 * 60  # how long should we stay up with no requests?
PORT = 8000
FAST_BOOT = False
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
VLLM_CACHE_PATH = "/root/.cache/vllm"
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)



vllm_image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "huggingface-hub==0.36.0",
    )
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1"
    })
)


app = modal.App(f"{MODEL_ID}_{GPU_TYPE}".replace("/", ".").replace("+", "p"))

@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_S,  # how long should we stay up with no requests?
    timeout=TIMEOUT_S,
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOL,
        VLLM_CACHE_PATH: VLLM_CACHE_VOL,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=MAX_NUM_SEQS
)
@modal.web_server(port=PORT, startup_timeout=TIMEOUT_S)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--uvicorn-log-level=info",
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        
        "--max-model-len", f"{MAX_MODEL_LEN}",
        "--gpu-memory-utilization", "0.93",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", f"{MAX_NUM_SEQS}",
        "--enable-prefix-caching",
        "--generation-config", "vllm",
        
        "--reasoning-parser", "qwen3",
        "--language-model-only",
        "--attention-backend", "FLASH_ATTN",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(*cmd)

    _ = subprocess.Popen(cmd)

