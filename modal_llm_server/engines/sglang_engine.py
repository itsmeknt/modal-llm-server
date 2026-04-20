import os
import subprocess
from typing import Any, ClassVar, Final, override

import modal

from modal_llm_server.config import Config
from modal_llm_server.engines.abstract_engine import AbstractSnapshottableEngine

class SGLangEngine(AbstractSnapshottableEngine):
    SGLANG_CACHE_PATH: ClassVar[str] = "/root/.cache/sglang"
    DEEP_GEMM_CACHE_PATH: ClassVar[str] = "/root/.cache/deep_gemm"
    TORCH_INDUCTOR_CACHE_PATH: ClassVar[str] = "/root/.cache/torch_inductor"

    sglang_cache_vol: Final[modal.Volume]
    deep_gemm_cache_vol: Final[modal.Volume]
    torch_inductor_cache_vol: Final[modal.Volume]
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        self.sglang_cache_vol = modal.Volume.from_name("sglang-cache", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes[SGLangEngine.SGLANG_CACHE_PATH] = self.sglang_cache_vol
        
        self.deep_gemm_cache_vol = modal.Volume.from_name("deep-gemm", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes[SGLangEngine.DEEP_GEMM_CACHE_PATH] = self.deep_gemm_cache_vol
        
        self.torch_inductor_cache_vol = modal.Volume.from_name("torch-inductor", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes[SGLangEngine.TORCH_INDUCTOR_CACHE_PATH] = self.torch_inductor_cache_vol

    @override
    def get_base_image(self) -> str:
        # WARNING(Kevin): none of these images currently work. sglang images is not stable for cu130 because it makes a call to cu129 which does not exist
        # return "lmsysorg/sglang:latest-cu130-runtime"
        return "lmsysorg/sglang:latest-runtime"
        # return "lmsysorg/sglang:v0.5.9-cu130-runtime"
        # return "lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1"
        

    @override
    def image_registry_kwargs(self) -> dict[str, Any]:
        return {}
        
    @override
    def extra_image_setup_commands(self) -> list[str]:
        return []
                
    @override
    def get_image_env_vars(self) -> dict[str, str]:
        return {
            # DeepGEMM
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "1",
            "SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS": "4",

            # lower-overhead Triton cache path used by SGLang on CUDA
            "SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE": "1",

            # snapshot compatibility and persistent torch.compile cache
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": SGLangEngine.TORCH_INDUCTOR_CACHE_PATH,
        }


    @override
    def prewarm_container(self) -> None:
        self._download_repo_snapshot()
        self._precompile_deepgemm()

    def _precompile_deepgemm(self) -> None:
        if os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1") != "1":
            return

        _ = subprocess.run(
            [
                "python3", "-m", "sglang.compile_deep_gemm",
                "--model-path", self.hf_cache_model_dir,
                "--tp", str(self.config.n_gpu),
            ],
            check=True,
        )


    @override
    def cmd(self) -> list[str]:
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.hf_cache_model_dir,
            "--served-model-name", self.config.served_model_name,
            "--host", "0.0.0.0",
            "--port", str(self.config.port),

            # parallelism / capacity
            "--tp", str(self.config.n_gpu),
            "--context-length", str(self.config.max_model_len),
            "--max-running-requests", str(self.config.max_num_seqs),

            # rough analogue to your vLLM gpu-memory-utilization tuning
            "--mem-fraction-static", "0.90",

            # required for SGLang snapshot sleep/resume flow
            "--enable-memory-saver",
            "--enable-weights-cpu-backup",

            # compile/capture more before snapshot
            "--cuda-graph-max-bs", str(self.config.max_num_seqs),
            "--piecewise-cuda-graph-max-tokens", "4096",
        ]

        repo = SGLangEngine.normalize(self.config.model_repo)

        # Reasoning parser support is documented by SGLang.
        if "qwen35" in repo or "qwen3" in repo:
            cmd.extend([
                "--reasoning-parser", "qwen3",
            ])

            # Only add a tool-call parser if you actually need tool calls.
            # cmd.extend(["--tool-call-parser", "qwen25"])

        # FP8 / NVFP4 backend pinning should be conditional on actual quant+GPU
        if "fp8" in repo:
            cmd += ["--fp8-gemm-backend", "auto"]
        if "nvfp4" in repo or "fp4" in repo:
            cmd += ["--fp4-gemm-backend", "auto"]    
            
        return cmd        
    
    @override
    def get_warmup_endpoint(self) -> str:
        return "/v1/chat/completions"

    @override
    def get_warmup_payload(self) -> dict[str, Any]:
        return {
            "model": self.config.served_model_name,
            "messages": [{"role": "user", "content": "Warmup"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    
    @override
    def get_health_check_endpoint(self) -> str | None:
        return f"/health"

    @override
    def get_blocked_admin_paths(self) -> set[str]:
        # minimum set for your public reverse proxy
        return {
            "release_memory_occupation",
            "resume_memory_occupation",
            "server_info",
        }

    @override
    def get_sleep_endpoint(self) -> str:
        return "/release_memory_occupation"

    @override
    def get_sleep_request_kwargs(self) -> dict[str, Any]:
        return {"json": {}}

    @override
    def get_wake_endpoint(self) -> str:
        return "/resume_memory_occupation"

    @override
    def get_wake_request_kwargs(self) -> dict[str, Any]:
        return {"json": {}}

