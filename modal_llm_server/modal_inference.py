# pyright: reportUnknownMemberType=false

"""
(One-time setup): Build image and download model files:

modal run modal_llm_server/modal_inference.py::prewarm_container

Then deploy via:

modal deploy modal_llm_server/modal_inference.py
"""

# TODO(Kevin): Modal single-GPU memory snapshots


from abc import ABC, abstractmethod
import asyncio
from collections.abc import Mapping
from pathlib import PurePosixPath
import subprocess
from typing import ClassVar, Final, override

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
import httpx
import modal


# MODEL CONFIG

#MODEL_REPO = "bartowski/Qwen_Qwen3.5-27B-GGUF"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q4_K_M"
#MODEL_FILE: str | None = "Qwen_Qwen3.5-27B-Q4_K_M.gguf"

#MODEL_REPO = "bartowski/Qwen_Qwen3.5-27B-GGUF"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q5_K_M"
#MODEL_FILE: str | None = "Qwen_Qwen3.5-27B-Q5_K_M.gguf"

#MODEL_REPO = "bartowski/Qwen_Qwen3.5-27B-GGUF"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q5_K_L"
#MODEL_FILE: str | None = "Qwen_Qwen3.5-27B-Q5_K_L.gguf"

#MODEL_REPO = "bartowski/Qwen_Qwen3.5-27B-GGUF"
#SERVED_MODEL_NAME = "bartowski/Qwen3.5-27B-Q6_K_L"
#MODEL_FILE: str | None = "Qwen_Qwen3.5-27B-Q6_K_L.gguf"



#MODEL_REPO = "google/gemma-4-31B-it"
#SERVED_MODEL_NAME = "google/gemma-4-31B-it-BF16"
#MODEL_FILE: str | None = None
#MAX_MODEL_LEN=128000
#MAX_NUM_SEQS=8


#MODEL_REPO = "nvidia/Gemma-4-31B-IT-NVFP4"
#SERVED_MODEL_NAME = "nvidia/Gemma-4-31B-IT-NVFP4"
#MODEL_FILE: str | None = None
#MAX_MODEL_LEN=128000
#MAX_NUM_SEQS=8



"""
MODEL_REPO = "Qwen/Qwen3.5-122B-A10B-FP8"
SERVED_MODEL_NAME = "Qwen/Qwen3.5-122B-A10B-FP8"
MODEL_FILE: str | None = None
MAX_MODEL_LEN=32000
MAX_NUM_SEQS=64
"""


"""
MODEL_REPO = "QuantTrio/Qwen3.5-122B-A10B-AWQ"
SERVED_MODEL_NAME = "QuantTrio/Qwen3.5-122B-A10B-AWQ"
MODEL_FILE: str | None = None
MAX_MODEL_LEN=32000
MAX_NUM_SEQS=64
"""

"""
MODEL_REPO = "Intel/Qwen3.5-122B-A10B-int4-AutoRound"
SERVED_MODEL_NAME = "Intel/Qwen3.5-122B-A10B-int4-AutoRound"
MODEL_FILE: str | None = None
MAX_MODEL_LEN=32000
MAX_NUM_SEQS=64
"""


"""
MODEL_REPO = "Sehyo/Qwen3.5-122B-A10B-NVFP4"
SERVED_MODEL_NAME = "Sehyo/Qwen3.5-122B-A10B-NVFP4"
MODEL_FILE: str | None = None
MAX_MODEL_LEN=32000
MAX_NUM_SEQS=64
"""



MODEL_REPO = "nvidia/Qwen3.5-397B-A17B-NVFP4"
SERVED_MODEL_NAME = "nvidia/Qwen3.5-397B-A17B-NVFP4"
MODEL_FILE: str | None = None
MAX_MODEL_LEN=32000
MAX_NUM_SEQS=32













# INFRA CONFIG
GPU_TYPE="B200+"
#GPU_TYPE="A100-80GB"
N_GPU=2
PREWARM_TIMEOUT_S = 60 * 60  # how long should we wait for the prewarm function to complete?
TIMEOUT_S = 20 * 60  # how long should we wait for container start?
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
HF_CACHE_MODEL_FILE = f"{HF_CACHE_MODEL_DIR}/{MODEL_FILE}" if MODEL_FILE is not None else ""
HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


def normalize(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())

class AbstractEngine(ABC):
    volumes: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]]

    def __init__(self):
        self.volumes = {
            HF_CACHE_ROOT_PATH: HF_CACHE_VOL,
        }

    @abstractmethod
    def get_base_image(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def prewarm_container(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def cmd(self) -> list[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_health_check_endpoint(self) -> str | None:
        raise NotImplementedError()
    
    def _download_model_file(self) -> None:
        """
        Downloads a single file from HuggingFace Hub, e.g. one quant from a GGUF repo
        """
        if MODEL_FILE is None:
            raise ValueError(f"MODEL_FILE must be specified to download a specific model file, but it was left as None!")
        
        from huggingface_hub import hf_hub_download    # pyright: ignore[reportUnknownVariableType]
        from pathlib import Path

        Path(HF_CACHE_MODEL_DIR).mkdir(parents=True, exist_ok=True)

        print(f"Downloading file {MODEL_REPO}/{MODEL_FILE} snapshot with requested destination {HF_CACHE_MODEL_DIR}...")
        local_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            # revision=MODEL_REVISION,
            local_dir=HF_CACHE_MODEL_DIR,
        )
        print(f"Downloading file {MODEL_REPO}/{MODEL_FILE} snapshot with requested destination {HF_CACHE_MODEL_DIR} done! Actual destination: {local_path}")
        HF_CACHE_VOL.commit()

    def _download_repo_snapshot(self) -> None:
        from huggingface_hub import snapshot_download    # pyright: ignore[reportUnknownVariableType]
        from pathlib import Path

        Path(HF_CACHE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading repo {MODEL_REPO} snapshot with requested destination {HF_CACHE_MODEL_DIR}...")
        local_path = snapshot_download(
            repo_id=MODEL_REPO,
            # revision=MODEL_REVISION,
            local_dir=HF_CACHE_MODEL_DIR,

            # Pull the multi-file checkpoint + config/tokenizer assets.
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.model",
                "*.tiktoken",
                "*.txt",
                "tokenizer*",
                "special_tokens_map.json",
                "generation_config.json",

                # Uncomment only if this repo needs remote code:
                # "*.py",
            ],
            ignore_patterns=[
                "*.bin",
                "*.pt",
                "*.onnx",
                "*.gguf",
            ],
        )
        print(f"Downloading repo {MODEL_REPO} snapshot with requested destination {HF_CACHE_MODEL_DIR} done! Actual destination: {local_path}")
        HF_CACHE_VOL.commit()

    
        

class VLLMEngine(AbstractEngine):
    VLLM_CACHE_PATH: ClassVar[str] = "/root/.cache/vllm"
    
    def __init__(self):
        super().__init__()
        
        VLLM_CACHE_VOL = modal.Volume.from_name("vllm-cache", create_if_missing=True)
        self.volumes[VLLMEngine.VLLM_CACHE_PATH] = VLLM_CACHE_VOL

    @override
    def get_base_image(self) -> str:
        repo = normalize(MODEL_REPO)
        #if "qwen35" in repo.lower():
        #    return "vllm/vllm-openai:qwen3_5-cu130"
        if "gemma4" in repo.lower():
            return "vllm/vllm-openai:gemma4-cu130"
        else:
            return "vllm/vllm-openai:cu130-nightly"

    @override
    def prewarm_container(self) -> None:
        self._download_repo_snapshot()
        
    @override
    def cmd(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            HF_CACHE_MODEL_DIR,
            "--uvicorn-log-level=info",
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),

            "--no-enforce-eager",
            "--max-model-len", f"{MAX_MODEL_LEN}",
            "--gpu-memory-utilization", "0.93",
            "--max-num-batched-tokens", "16384",
            "--max-num-seqs", f"{MAX_NUM_SEQS}",
            "--enable-prefix-caching",
            "--generation-config", "vllm",
            "--tensor-parallel-size", str(N_GPU),
        ]
        
        repo = normalize(MODEL_REPO).lower()
        if "qwen35" in repo:
            cmd.extend([
                "--reasoning-parser", "qwen3",
                "--language-model-only",
                # "--enable-auto-tool-choice",
                # "--tool-call-parser", "qwen3_coder",
                # "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":1}",
            ])

            if "nvfp4" not in repo:
                # MTP seems to break with nvfp4, so only add it if not
                cmd.extend([
                    "--speculative-config.method", "mtp",
                    "--speculative-config.num_speculative_tokens", "1",
                ])

            
            """
            # NOTE(Kevin): vllm bug with Qwen3.5 models, need to force
            # FLASH_ATTN and language-model-only or the quality degrades
            if "27b" in repo:
                cmd.extend([
                     "--attention-backend", "FLASH_ATTN"
                ])
            """

            if "intel" in repo and "int4autoround" in repo:
                cmd.extend([
                    "--tokenizer", "Qwen/Qwen3.5-122B-A10B",
                ])
        elif "gemma4" in repo:
            cmd.extend([
                "--reasoning-parser", "gemma4",
                 "--language-model-only",
            ])

        return cmd

    @override
    def get_health_check_endpoint(self) -> str | None:
        return f"http://127.0.0.1:{PORT}/health"
            
class LlamaCPPEngine(AbstractEngine):
    @override
    def get_base_image(self) -> str:
        return "ghcr.io/ggml-org/llama.cpp:server-cuda13"
    
    @override
    def prewarm_container(self) -> None:
        self._download_model_file()
        
    
    @override
    def cmd(self) -> list[str]:
        if MODEL_FILE is None:
            raise ValueError(f"MODEL_FILE must be specified to run llama-server, but it was left as None!")
        
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
        ]
    
        repo = normalize(MODEL_REPO)
        if "qwen35" in repo.lower():
            # NOTE(Kevin): llama.cpp bug with Qwen3.5 -- client-side sampling
            # params are not respected, so have to set these server side
            cmd.extend([
                "--temp", "0.7",
                "--top-p", "0.8",
                "--top-k", "20",
                "--min-p", "0.0",
                "--presence-penalty", "1.5",
                "--repeat-penalty", "1.0",
                "--chat-template-kwargs", '{"enable_thinking": false}',
            ])

        return cmd
            

    @override
    def get_health_check_endpoint(self) -> str | None:
        return f"http://127.0.0.1:{PORT}/health"
    
engine = VLLMEngine()

image = (
    modal.Image.from_registry(
        engine.get_base_image(),
        add_python="3.12",    # NOTE(Kevin): this is the container Python environment. VLLM/SGLang/llama.cpp likely have their own Python environment and need to pip install it there
    )
    .entrypoint([])
    .uv_pip_install(
        "huggingface_hub>=1.0",
        # "requests==2.33.0",
        "fastapi>=0.115",
        "flashinfer-python==0.6.7.post3",
        "flashinfer-cubin==0.6.7.post3",
        "httpx>=0.27",
    )
    .run_commands(
        "python -m pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130"
    )
    .env({
        "HF_HOME": HF_CACHE_ROOT_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_ROOT_PATH}/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_CACHE_ROOT": VLLMEngine.VLLM_CACHE_PATH,
    })
)


app = modal.App(f"{SERVED_MODEL_NAME}_{GPU_TYPE}".replace("/", ".").replace("+", "p"))

@app.function(
    image=image,
    volumes=engine.volumes,
    timeout=PREWARM_TIMEOUT_S,
)
def prewarm_container():
    engine.prewarm_container()


@app.cls(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    volumes=engine.volumes,
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class Serve:
    cmd: list[str]
    proc: subprocess.Popen    # pyright: ignore[reportMissingTypeArgument]
    upstream: str
    client: httpx.AsyncClient
    
    @modal.enter()
    async def start_engine(self):
        self.cmd = engine.cmd()
        print("Starting:", " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd)

        self.upstream = f"http://127.0.0.1:{PORT}"
        self.client = httpx.AsyncClient(
            base_url=self.upstream,
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=MAX_NUM_SEQS,
                max_connections=max(MAX_NUM_SEQS * 2, 32),
            ),
        )

        health_check_endpoint = engine.get_health_check_endpoint()
        if health_check_endpoint is not None:
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{engine.__class__.__name__} exited early with code {self.proc.returncode}"
                    )

                try:
                    print(
                        f"Polling {engine.__class__.__name__} health check endpoint at {health_check_endpoint}..."
                    )
                    r = await self.client.get(health_check_endpoint, timeout=2.0)
                    if r.status_code == 200:
                        print(f"{engine.__class__.__name__} is healthy!")
                        break
                except Exception:
                    pass

                await asyncio.sleep(4)

    @modal.exit()
    async def stop_engine(self):
        if hasattr(self, "client"):
            await self.client.aclose()

        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            try:
                _ = await asyncio.to_thread(self.proc.wait, 10)
            except Exception:
                self.proc.kill()

                _ = await asyncio.to_thread(self.proc.wait, 10)

    @modal.asgi_app()
    def app(self):
        api = FastAPI()

        hop_by_hop_headers = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "host",
            "content-length",
        }

        def filter_headers(headers: Mapping[str, str]) -> dict[str, str]:
            return {
                k: v
                for k, v in headers.items()
                if k.lower() not in hop_by_hop_headers
            }

        @api.api_route(
            "/",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        )
        @api.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        )
        async def proxy(request: Request, path: str = "") -> Response:    # pyright: ignore[reportUnusedFunction]
            if not hasattr(self, "proc"):
                return JSONResponse(
                    {"error": f"{engine.__class__.__name__} has not yet initialized"},
                    status_code=503,
                )
                
            if self.proc.poll() is not None:
                return JSONResponse(
                    {"error": f"{engine.__class__.__name__} is not running"},
                    status_code=503,
                )

            upstream_path = f"/{path}" if path else "/"
            body = await request.body()

            upstream_request = self.client.build_request(
                method=request.method,
                url=upstream_path,
                params=request.query_params,
                headers=filter_headers(request.headers),
                content=body,
            )

            try:
                upstream_response = await self.client.send(upstream_request, stream=True)   
                response_headers = filter_headers(upstream_response.headers)
                return StreamingResponse(
                    upstream_response.aiter_raw(),
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                    background=BackgroundTask(upstream_response.aclose),
                )
            except httpx.RequestError as e:
                return JSONResponse(
                    {"error": f"Upstream request failed: {type(e).__name__}: {e}"},
                    status_code=503,
                )


        return api


"""
@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_S,
    timeout=TIMEOUT_S,
    volumes=engine.volumes,
)
@modal.web_server(port=PORT, startup_timeout=TIMEOUT_S)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
def serve():
    import requests
    import subprocess
    import time


    cmd = engine.cmd()
    print("Starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    health_check_endpoint = engine.get_health_check_endpoint()
    if health_check_endpoint is not None:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{engine.__class__.__name__} exited early with code {proc.returncode}")

            try:
                print(f"Polling {engine.__class__.__name__} health check endpoint at {health_check_endpoint}...")
                r = requests.get(health_check_endpoint, timeout=2)
                if r.status_code == 200:
                    print(f"{engine.__class__.__name__} is healthy!")
                    return
            except Exception:
                pass

            time.sleep(5)

"""
