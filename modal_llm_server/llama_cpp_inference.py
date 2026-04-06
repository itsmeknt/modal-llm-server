# pyright: reportUnknownMemberType=false

"""
(One-time setup): Download model files:

modal run modal_llm_server/llama_cpp_inference.py::download_model

Then deploy via:

modal deploy modal_llm_server/llama_cpp_inference.py
"""

import modal

#MODEL_REPO = "bartowski/Qwen_Qwen3.5-27B-GGUF"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q4_K_M"
#MODEL_FILE = "Qwen_Qwen3.5-27B-Q4_K_M.gguf"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q5_K_M"
#MODEL_FILE = "Qwen_Qwen3.5-27B-Q5_K_M.gguf"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q5_K_L"
#MODEL_FILE = "Qwen_Qwen3.5-27B-Q5_K_L.gguf"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q6_K_L"
#MODEL_FILE = "Qwen_Qwen3.5-27B-Q6_K_L.gguf"



MODEL_REPO = "unsloth/Qwen3.5-27B-GGUF"
SERVED_MODEL_NAME = "unsloth/Qwen3.5-27B-UD-Q6_K_XL"
MODEL_FILE = "Qwen3.5-27B-UD-Q6_K_XL.gguf"



# MODEL_REVISION = "b7ca741b86de18df552fd2cc952861e04621a4bd"



IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda13"
GPU_TYPE="A100-80GB"
N_GPU=1
MAX_MODEL_LEN=16000
MAX_NUM_SEQS=8


TIMEOUT_S = 10 * 60  # how long should we wait for container start?
SCALEDOWN_S = 1 * 60  # how long should we stay up with no requests?
PORT = 8000

"""
We want the cache file structure to look something like this:
/root/.cache/huggingface/
├── hub/   # regular HF hub cache root for other hub operations
└── bartowski/
    └── Qwen_Qwen3.5-27B-GGUF/
        ├── Qwen3.5-27B-Q4_K_M.gguf
        └── .cache/huggingface/
"""
HF_CACHE_ROOT_PATH = f"/root/.cache/huggingface"
HF_CACHE_MODEL_DIR = f"{HF_CACHE_ROOT_PATH}/{MODEL_REPO}"
HF_CACHE_MODEL_FILE = f"{HF_CACHE_MODEL_DIR}/{MODEL_FILE}"
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


image = (
    modal.Image.from_registry(
        IMAGE,
        add_python="3.11",
    )
    .entrypoint([])
    .uv_pip_install(
        "huggingface_hub>=1.0",
        "requests==2.33.0",
    )
    .env({
        "HF_HOME": HF_CACHE_ROOT_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_ROOT_PATH}/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)


app = modal.App(f"{MODEL_FILE}_{GPU_TYPE}".replace("/", ".").replace("+", "p"))

@app.function(
    image=image,
    volumes={HF_CACHE_ROOT_PATH: HF_CACHE_VOL},
    timeout=60 * 60,
)
def download_model():
    """
    Run this once to populate the Modal Volume with the GGUF via:
    
    modal run modal_llama_cpp.py::download_model
    """
    from huggingface_hub import hf_hub_download    # pyright: ignore[reportUnknownVariableType]
    from pathlib import Path

    Path(HF_CACHE_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        # revision=MODEL_REVISION,
        local_dir=HF_CACHE_MODEL_DIR,
    )
    print(f"Downloaded model to: {local_path}")

    # Persist changes so later containers can see the file
    HF_CACHE_VOL.commit()



@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    volumes={
        HF_CACHE_ROOT_PATH: HF_CACHE_VOL,
    },
)
@modal.web_server(port=PORT, startup_timeout=30 * 60)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
def serve():
    """
    Start llama-server as the container's HTTP server.
    """
    import requests
    import subprocess
    import time

    cmd = [
        "/app/llama-server",
        "-m", HF_CACHE_MODEL_FILE,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--alias", SERVED_MODEL_NAME,
        "--jinja",
        "--ctx-size", str(MAX_MODEL_LEN * MAX_NUM_SEQS),
        "--n-gpu-layers", "all",
        "--parallel", str(MAX_NUM_SEQS),
        "--flash-attn", "on",
        "--no-mmproj",
        "-b", "4096",
        "-ub", "1024",
        "--temp", "0.7",
        "--top-p", "0.8",
        "--top-k", "20",
        "--min-p", "0.0",
        "--presence-penalty", "1.5",
        "--repeat-penalty", "1.0",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ]

    print("Starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    # Wait for llama-server healthcheck in a forever loop so startup failures are obvious.
    url = f"http://127.0.0.1:{PORT}/health"
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")

        try:
            print("Testing llama-server health check...")
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print("llama-server is healthy")
                return
        except Exception:
            pass

        time.sleep(2)

