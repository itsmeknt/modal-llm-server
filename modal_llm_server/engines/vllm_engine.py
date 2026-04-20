from typing import Any, ClassVar, Final, override

import modal

from modal_llm_server.config import Config
from modal_llm_server.engines.abstract_engine import AbstractSnapshottableEngine

class VLLMEngine(AbstractSnapshottableEngine):
    VLLM_CACHE_PATH: ClassVar[str] = "/root/.cache/vllm"
    TORCH_INDUCTOR_CACHE_PATH: ClassVar[str] = "/root/.cache/torch_inductor"
    
    vllm_cache_vol: Final[modal.Volume]
    torch_inductor_cache_vol: Final[modal.Volume]
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        self.vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes[VLLMEngine.VLLM_CACHE_PATH] = self.vllm_cache_vol
        
        self.torch_inductor_cache_vol = modal.Volume.from_name("torch-inductor", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes[VLLMEngine.TORCH_INDUCTOR_CACHE_PATH] = self.torch_inductor_cache_vol

    @override
    def get_base_image(self) -> str:
        repo = VLLMEngine.normalize(self.config.model_repo)
        #if "qwen35" in repo.lower():
        #    return "vllm/vllm-openai:qwen3_5-cu130"
        if "gemma4" in repo.lower():
            return "vllm/vllm-openai:gemma4-cu130"
        else:
            return "vllm/vllm-openai:cu130-nightly"

    @override
    def image_registry_kwargs(self) -> dict[str, Any]:
        return {"add_python": "3.12"}
        
    @override
    def extra_image_setup_commands(self) -> list[str]:
        return [
            "python3 -m pip install --no-cache-dir 'flashinfer-python==0.6.7.post3' 'flashinfer-cubin==0.6.7.post3'",
            "python3 -m pip install --no-cache-dir 'flashinfer-jit-cache' --index-url https://flashinfer.ai/whl/cu130",
        ]

    @override
    def get_image_env_vars(self) -> dict[str, str]:
        return {
            "VLLM_CACHE_ROOT": VLLMEngine.VLLM_CACHE_PATH,
            
            # required for /sleep and /wake_up
            "VLLM_SERVER_DEV_MODE": "1",
            
            # snapshot compatibility and persistent torch.compile cache
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": VLLMEngine.TORCH_INDUCTOR_CACHE_PATH,
        }
    
    @override
    def prewarm_container(self) -> None:
        self._download_repo_snapshot()
        
    @override
    def cmd(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.hf_cache_model_dir,
            "--uvicorn-log-level=info",
            "--served-model-name", self.config.served_model_name,
            "--host", "0.0.0.0",
            "--port", str(self.config.port),

            "--no-enforce-eager",
            "--max-model-len", f"{self.config.max_model_len}",
            "--gpu-memory-utilization", "0.93",
            "--max-num-batched-tokens", "8192",
            # "--max-num-batched-tokens", "16384",
            "--max-num-seqs", f"{self.config.max_num_seqs}",
            "--enable-prefix-caching",
            "--generation-config", "vllm",
            "--tensor-parallel-size", str(self.config.n_gpu),
        ]
        
        repo = VLLMEngine.normalize(self.config.model_repo)
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
        return {
          "sleep",
          "wake_up",
          "is_sleeping",
          "collective_rpc",
          "server_info",
          "reset_prefix_cache",
          "reset_mm_cache",
          "reset_encoder_cache",
      }
    
    @override
    def get_sleep_endpoint(self) -> str:
        return "/sleep"

    @override
    def get_sleep_request_kwargs(self) -> dict[str, Any]:
        return {"params": {"level": 1}}
    
    @override
    def get_wake_endpoint(self) -> str:
        return "/wake_up"

    @override
    def get_wake_request_kwargs(self) -> dict[str, Any]:
        return {}
