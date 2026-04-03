# pyright: reportUnknownMemberType=false

"""
Deploy via: modal deploy sglang_inference.py
"""


import subprocess
import time

import modal



MODEL_ID = "Qwen/Qwen3.5-27B"
SERVED_MODEL_NAME = f"{MODEL_ID}-BF16"
MODEL_REVISION = "b7ca741b86de18df552fd2cc952861e04621a4bd"  # 2026/02/25


IMAGE="lmsysorg/sglang:nightly-dev-20260327-8a4cdcd5"
# IMAGE="lmsysorg/sglang:nightly-dev-cu13-20260327-8a4cdcd5"

# GPU_TYPE="B200+"
GPU_TYPE="H200"
# GPU_TYPE="A100-80GB"
N_GPU=1
MAX_MODEL_LEN=16000
MAX_RUNNING_REQS=90



TIMEOUT_S = 10 * 60  # how long should we wait for container start?
SCALEDOWN_S = 5 * 60  # how long should we stay up with no requests?
PORT = 8000
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"
DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)



def compile_deep_gemm():
    import os

    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        _ = subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_ID} --revision {MODEL_REVISION} --tp {N_GPU}",
            shell=True,
        )
        
sglang_image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .uv_pip_install(
        "huggingface-hub==1.8.0"
    )
    .env({
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1"
    })
    .run_function(
        compile_deep_gemm,
        volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
        gpu=f"{GPU_TYPE}:{N_GPU}",
    )
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
)

   

app = modal.App(f"{MODEL_ID}_{GPU_TYPE}".replace("/", ".").replace("+", "p"))

@app.function(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_S,  # how long should we stay up with no requests?
    timeout=TIMEOUT_S,
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOL,
        DG_CACHE_PATH: DG_CACHE_VOL
    },
)
@modal.concurrent(max_inputs=MAX_RUNNING_REQS)
@modal.web_server(port=PORT, startup_timeout=TIMEOUT_S)
def serve():
    import subprocess

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        f"{PORT}",

        "--tp",  # use all GPUs to split up tensor-parallel operations
        f"{N_GPU}",
        "--context-length",
        F"{MAX_MODEL_LEN}",
        "--cuda-graph-max-bs",  # capture CUDA graphs up to batch sizes we're likely to observe
        f"{MAX_RUNNING_REQS}",
        "--max-running-requests",
        f"{MAX_RUNNING_REQS}",
        "--mem-fraction-static", "0.88",

        "--reasoning-parser", "qwen3",
        # "--attention-backend", "trtllm_mha",
        "--mamba-ssm-dtype", "bfloat16",
        "--mamba-full-memory-ratio", "0.7",
        "--disable-cuda-graph",
        # "--mamba-scheduler-strategy", "extra_buffer",
        "--page-size", "64",
        
        "--kv-cache-dtype", "bf16",
        "--disable-radix-cache",
        "--mamba-scheduler-strategy", "no_buffer",
        
        "--enable-metrics",  # expose metrics endpoints for telemetry
        "--enable-memory-saver",  # enable offload, for snapshotting
        "--enable-weights-cpu-backup",  # enable offload, for snapshotting
    ]

    print(*cmd)
    
    _ = subprocess.Popen(cmd)
