# NOTE(Kevin): untested and likely buggy

import json
import textwrap
from pathlib import PurePosixPath
from typing import Any, ClassVar, override

from modal_llm_server.config import Config
from modal_llm_server.engines.abstract_engine import AbstractEngine


class TabbyEngine(AbstractEngine):
    _KV_CACHE_MODE: ClassVar[str] = "8,8"
    _CACHE_SIZE_MULTIPLE: ClassVar[int] = 256

    def __init__(self, config: Config):
        super().__init__(config)

    @override
    def get_base_image(self) -> str:
        return "ghcr.io/theroyallab/tabbyapi:latest"

    @override
    def image_registry_kwargs(self) -> dict[str, Any]:
        return {}

    @override
    def extra_image_setup_commands(self) -> list[str]:
        return []

    @override
    def get_image_env_vars(self) -> dict[str, str]:
        return {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NAME": "TabbyAPI",
        }

    @override
    def prewarm_container(self) -> None:
        self._download_repo_snapshot()

    @override
    def cmd(self) -> list[str]:
        config_yml = self._get_tabby_config_yml().rstrip()

        return [
            "/bin/bash",
            "-lc",
            (
                "cat > /app/config.yml <<'YAML'\n"
                f"{config_yml}\n"
                "YAML\n"
                "exec /opt/venv/bin/python /app/main.py"
            ),
        ]

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
        return "/health"

    def _get_tabby_config_yml(self) -> str:
        cache_size = self._round_up_to_multiple(
            value=self.config.max_model_len * self.config.max_num_seqs,
            multiple=TabbyEngine._CACHE_SIZE_MULTIPLE,
        )

        model_dir_path = PurePosixPath(self.hf_cache_model_dir)
        model_dir = self._yaml_string(str(model_dir_path.parent))
        model_name = self._yaml_string(str(model_dir_path.name))
        served_model_name = self._yaml_string(self.config.served_model_name)

        reasoning = "false"
        reasoning_start_token = ""
        reasoning_end_token = ""
        tool_format = ""
        
        repo = TabbyEngine.normalize(self.config.model_repo)
        if "qwen35" in repo:
            reasoning = "true"
            reasoning_start_token = '"<think>"'
            reasoning_end_token = '"</think>"'
            tool_format = "qwen3_5"
            
        return textwrap.dedent(
            f"""\
            network:
              host: "0.0.0.0"
              port: {self.config.port}
              disable_auth: true
              disable_fetch_requests: true
              send_tracebacks: false
              api_servers: ["OAI"]

            logging:
              log_prompt: false
              log_generation_params: false
              log_requests: false

            model:
              model_dir: {model_dir}
              inline_model_loading: false
              use_dummy_models: true
              dummy_model_names: [{served_model_name}]
              model_name: {model_name}
              use_as_default:
                - "max_seq_len"
                - "cache_size"
                - "cache_mode"
                - "chunk_size"
                - "max_batch_size"

              # EXL2 path.
              backend: "exllamav3"

              # Per-request context length.
              max_seq_len: {self.config.max_model_len}

              # Global KV cache budget. Rounded to TabbyAPI's required multiple.
              cache_size: {cache_size}

              # FP16 is safest; Q8 usually gives a better concurrency/VRAM tradeoff.
              cache_mode: "{TabbyEngine._KV_CACHE_MODE}"

              # Prefer autosplit unless you have a known multi-GPU split.
              tensor_parallel: {"true" if self.config.n_gpu > 1 else "false"}
              tensor_parallel_backend: "native"
              gpu_split_auto: true
              autosplit_reserve: [96]
              gpu_split: []

              rope_scale: 1.0
              rope_alpha:

              chunk_size: 2048
              output_chunking: true
              max_batch_size: {self.config.max_num_seqs}

              # Empty prompt_template lets TabbyAPI use the model tokenizer config.
              prompt_template:

              vision: false
              force_enable_thinking: false
            
              reasoning: {reasoning}
              reasoning_start_token: {reasoning_start_token}
              reasoning_end_token: {reasoning_end_token}
              reasoning_suppress_header:
            
              tool_format: {tool_format}

            sampling:
              override_preset:

            lora:
              lora_dir: "loras"
              loras:

            embeddings:
              embedding_model_dir: "models"
              embeddings_device: "cuda"
              embedding_model_name:

            memory:
              sysmem_recurrent_cache: 0
              cuda_malloc_async: true

            developer:
              unsafe_launch: false
              disable_request_streaming: false
              realtime_process_priority: false
              seqlog: false
              seqlog_server_url: "http://localhost:5341"
              seqlog_api_key:
            """
        )

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple}")

        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _yaml_string(value: str) -> str:
        # YAML accepts JSON string syntax, and this avoids quote/escape edge cases.
        return json.dumps(value)
