from typing import Any, override


from modal_llm_server.config import Config
from modal_llm_server.engines.abstract_engine import AbstractEngine

class LlamaCPPEngine(AbstractEngine):
    def __init__(self, config: Config):
        super().__init__(config)

    @override
    def get_base_image(self) -> str:
        return "ghcr.io/ggml-org/llama.cpp:server-cuda13"

    @override
    def image_registry_kwargs(self) -> dict[str, Any]:
        return {"add_python": "3.12"}
        
    @override
    def extra_image_setup_commands(self) -> list[str]:
        return []

    @override
    def get_image_env_vars(self) -> dict[str, str]:
        return {}

    @override
    def prewarm_container(self) -> None:
        self._download_model_file()

    @override
    def cmd(self) -> list[str]:
        if self.config.model_file is None:
            raise ValueError(f"self.config.model_file must be specified to run llama-server, but it was left as None!")
        
        cmd = [
            "/app/llama-server",
            "-m", self.hf_cache_model_file,
            "--host", "0.0.0.0",
            "--port", str(self.config.port),
            "--alias", self.config.served_model_name,
            "--jinja",
            "--ctx-size", str(self.config.max_model_len * self.config.max_num_seqs),
            "--n-gpu-layers", "all",
            "--parallel", str(self.config.max_num_seqs),
            "--flash-attn", "on",
            "--no-mmproj",
            "-b", "4096",
            "-ub", "1024",
        ]
    
        repo = LlamaCPPEngine.normalize(self.config.model_repo)
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
    
