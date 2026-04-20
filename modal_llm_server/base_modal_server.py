# pyright: reportUnknownMemberType=false

import asyncio
import copy
import subprocess
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
import modal
from starlette.background import BackgroundTask

from modal_llm_server.config import Config, get_active_config, Globals
from modal_llm_server.engines.llama_cpp_engine import LlamaCPPEngine
from modal_llm_server.engines.sglang_engine import SGLangEngine
from modal_llm_server.engines.vllm_engine import VLLMEngine


def build_engine(config: Config):
    if config.engine == "vllm":
        return VLLMEngine(config)
    if config.engine == "sglang":
        return SGLangEngine(config)
    if config.engine == "llamacpp":
        return LlamaCPPEngine(config)
    raise NotImplementedError(f"Config engine {config.engine} not yet implemented!")

CONFIG = get_active_config()
ENGINE = build_engine(CONFIG)

IMAGE = (
    modal.Image.from_registry(
        ENGINE.get_base_image(),
        **ENGINE.image_registry_kwargs(),
    )
    .entrypoint([])
    .run_commands(
        "echo \"Trying PYTHON_BIN=python first...\"; python -m pip install --no-cache-dir 'huggingface_hub>=1.0' 'fastapi>=0.115' 'httpx>=0.27' || { echo \"PYTHON_BIN=python failed, trying PYTHON_BIN=python3 next...\"; python3 -m pip install --no-cache-dir 'huggingface_hub>=1.0' 'fastapi>=0.115' 'httpx>=0.27'; }",    # These are for setting up the Modal async HTTP server
        *ENGINE.extra_image_setup_commands(),    # Any extra dependencies for the engine itself goes here, e.g. flashinfer
    )
    .env({
        **ENGINE.get_hf_env_vars(),
        **ENGINE.get_image_env_vars(),
    })
)

def app_name(config: Config) -> str:
    return f"{config.served_model_name}_{config.n_gpu}x{config.gpu_type}_{ENGINE.__class__.__name__}".replace("/", ".").replace("+", "p")


class ServeBase:
    cmd: list[str]
    proc: subprocess.Popen[bytes]
    upstream: str
    client: httpx.AsyncClient

    def new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.upstream,
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=ENGINE.config.max_num_seqs,
                max_connections=max(ENGINE.config.max_num_seqs * 2, 32),
            ),
        )

    async def wait_ready(self) -> None:
        health_check_endpoint = ENGINE.get_health_check_endpoint()
        if health_check_endpoint is None:
            return

        while True:
            if hasattr(self, "proc") and self.proc.poll() is not None:
                raise RuntimeError(
                    f"{ENGINE.__class__.__name__} exited early with code {self.proc.returncode}"
                )

            try:
                print(f"Polling {ENGINE.__class__.__name__} health check endpoint at {health_check_endpoint}...")
                r = await self.client.get(
                    f"http://127.0.0.1:{ENGINE.config.port}{health_check_endpoint}",
                    timeout=2.0,
                )
                if r.status_code == 200:
                    print(f"{ENGINE.__class__.__name__} is healthy!")
                    return
            except Exception:
                pass

            await asyncio.sleep(4)

    async def _warmup(self) -> None:
        for _ in range(2):
            r = await self.client.post(
                ENGINE.get_warmup_endpoint(),
                json=ENGINE.get_warmup_payload(),
                timeout=180.0,
            )
            _ = r.raise_for_status()
            _ = r.json()

    async def start_engine_base(self) -> None:
        self.cmd = ENGINE.cmd()
        self.upstream = f"http://127.0.0.1:{ENGINE.config.port}"
        
        print("Starting:", " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd)
        
        self.client = self.new_client()
        await self.wait_ready()
        await self._warmup()

    async def stop_engine_base(self) -> None:
        if hasattr(self, "client"):
            await self.client.aclose()

        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            try:
                _ = await asyncio.to_thread(self.proc.wait, 10)
            except Exception:
                self.proc.kill()
                
                _ = await asyncio.to_thread(self.proc.wait, 10)

    def _is_public_route(self, method: str, path: str) -> bool:
        health_check_endpoint = ENGINE.get_health_check_endpoint()
        return (
            health_check_endpoint is not None
            and path == health_check_endpoint
            and method.upper() in {"GET", "HEAD"}
        )

    def _is_allowed_route(self, method: str, path: str) -> bool:
        if self._is_public_route(method, path):
            return True

        allowed_methods = Globals.ALLOWED_ROUTES.get(path)
        return allowed_methods is not None and method.upper() in allowed_methods

    def _is_authorized(self, request: Request, request_path: str) -> bool:
        if self._is_public_route(request.method, request_path):
            return True

        if not Globals.API_KEYS:
            return True

        bearer_keys = {f"Bearer {api_key}" for api_key in Globals.API_KEYS}
        return request.headers.get("authorization") in bearer_keys                

    def _filter_headers(self, headers: Mapping[str, str]) -> dict[str, str]:   
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
    
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in hop_by_hop_headers
        }    

    @modal.asgi_app()
    def app(self) -> FastAPI:
        api = FastAPI()

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
                    {"error": f"{ENGINE.__class__.__name__} has not yet initialized"},
                    status_code=503,
                )
                
            if self.proc.poll() is not None:
                return JSONResponse(
                    {"error": f"{ENGINE.__class__.__name__} is not running"},
                    status_code=503,
                )

            request_path = "/" + path.lstrip("/")
            
            if not self._is_authorized(request, request_path):
                return JSONResponse({"error": "unauthorized"}, status_code=401)

            if not self._is_allowed_route(request.method, request_path):
                return JSONResponse({"error": "not found"}, status_code=404)
            
            body = await request.body()
            
            upstream_request = self.client.build_request(
                method=request.method,
                url=request_path,
                params=request.query_params,
                headers=self._filter_headers(request.headers),
                content=body,
            )

            try:
                upstream_response = await self.client.send(upstream_request, stream=True)   
                response_headers = self._filter_headers(upstream_response.headers)
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

