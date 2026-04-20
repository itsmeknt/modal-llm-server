# pyright: reportUnknownMemberType=false

"""
(One-time setup): Build image and download model files:

modal run modal_llm_server/modal_snapshot_inference.py::prewarm_container

Then deploy via:

modal deploy modal_llm_server/modal_snapshot_inference.py
"""

# TODO(Kevin): Modal single-GPU memory snapshots


from abc import ABC, abstractmethod
import asyncio
from collections.abc import Mapping
import os
from pathlib import PurePosixPath
import subprocess
from typing import Any, ClassVar, Final, override

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
import httpx
import modal

from modal_llm_server.config import get_active_config, Globals
from modal_llm_server.engines.llama_cpp_engine import LlamaCPPEngine
from modal_llm_server.engines.sglang_engine import SGLangEngine
from modal_llm_server.engines.vllm_engine import VLLMEngine




CONFIG = get_active_config()
if CONFIG.engine == "vllm":
    engine = VLLMEngine(CONFIG)
elif CONFIG.engine == "sglang":
    engine = SGLangEngine(CONFIG)
elif CONFIG.engine == "llamacpp":
    raise RuntimeError(f"llamacpp not supported for snapshottable Modal instance!")
else:
    raise NotImplementedError(f"Config engine {CONFIG.engine} not yet implemented!")



image = (
    modal.Image.from_registry(
        engine.get_base_image(),
        **engine.image_registry_kwargs(),
    )
    .entrypoint([])
    .run_commands(
        "echo \"Trying PYTHON_BIN=python first...\"; python -m pip install --no-cache-dir 'huggingface_hub>=1.0' 'fastapi>=0.115' 'httpx>=0.27' || { echo \"PYTHON_BIN=python failed, trying PYTHON_BIN=python3 next...\"; python3 -m pip install --no-cache-dir 'huggingface_hub>=1.0' 'fastapi>=0.115' 'httpx>=0.27'; }",    # These are for setting up the Modal async HTTP server
        *engine.extra_image_setup_commands(),    # Any extra dependencies for the engine itself goes here, e.g. flashinfer
    )
    .env({
        **engine.get_hf_env_vars(),
        **engine.get_image_env_vars(),
    })
)

app = modal.App(f"{CONFIG.served_model_name}_{CONFIG.gpu_type}x{CONFIG.n_gpu}_{engine.__class__.__name__}".replace("/", ".").replace("+", "p"))


@app.function(
    image=image,
    volumes=engine.volumes,
    timeout=Globals.PREWARM_TIMEOUT_S,
)
def prewarm_container():
    engine.prewarm_container()


@app.cls(
    image=image,
    gpu=f"{engine.config.gpu_type}:{engine.config.n_gpu}",
    scaledown_window=Globals.SCALEDOWN_S,
    timeout=Globals.TIMEOUT_S,
    volumes=engine.volumes,
    
    # turn on snapshots
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=engine.config.max_num_seqs)
class Serve:
    cmd: list[str]
    proc: subprocess.Popen    # pyright: ignore[reportMissingTypeArgument]
    upstream: str
    client: httpx.AsyncClient

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.upstream,
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=engine.config.max_num_seqs,
                max_connections=max(engine.config.max_num_seqs * 2, 32),
            ),
        )

    async def _wait_ready(self) -> None:
        health_check_endpoint = engine.get_health_check_endpoint()
        if health_check_endpoint is None:
            return

        while True:
            if hasattr(self, "proc") and self.proc.poll() is not None:
                raise RuntimeError(
                    f"{engine.__class__.__name__} exited early with code {self.proc.returncode}"
                )

            try:
                print(f"Polling {engine.__class__.__name__} health check endpoint at {health_check_endpoint}...")
                r = await self.client.get(f"http://127.0.0.1:{engine.config.port}{health_check_endpoint}", timeout=2.0)
                if r.status_code == 200:
                    print(f"{engine.__class__.__name__} is healthy!")
                    return
            except Exception:
                pass

            await asyncio.sleep(4)

    async def _warmup(self) -> None:
        for _ in range(2):
            r = await self.client.post(engine.get_warmup_endpoint(), json=engine.get_warmup_payload(), timeout=180.0)
            _ = r.raise_for_status()
            _ = r.json()
            
    async def _sleep_server(self) -> None:
        r = await self.client.post(engine.get_sleep_endpoint(), **engine.get_sleep_request_kwargs(), timeout=300.0)
        _ = r.raise_for_status()

    async def _wake_server(self) -> None:
        r = await self.client.post(engine.get_wake_endpoint(), **engine.get_wake_request_kwargs(), timeout=300.0)
        _ = r.raise_for_status()
        await self._wait_ready()
        
    @modal.enter(snap=True)
    async def start_engine_and_snapshot(self):
        self.cmd = engine.cmd()
        self.upstream = f"http://127.0.0.1:{engine.config.port}"

        print("Starting:", " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd)
        
        self.client = self._new_client()
        await self._wait_ready()
        await self._warmup()
        
        await self._sleep_server()
        # do not carry a live client socket pool into restore
        await self.client.aclose()
        
    @modal.enter(snap=False)
    async def wake_after_restore(self):
        self.upstream = f"http://127.0.0.1:{engine.config.port}"
        self.client = self._new_client()
        await self._wake_server()
        

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

            if path.lstrip("/") in engine.get_blocked_admin_paths():
                return JSONResponse({"error": "not found"}, status_code=404)
            
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

