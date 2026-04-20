from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import Any, Final

import modal

from modal_llm_server.config import Config, Globals



class AbstractEngine(ABC):
    config: Final[Config]
    hf_cache_model_dir: Final[str]
    hf_cache_model_file: Final[str]

    hf_cache_vol: Final[modal.Volume]
    volumes: Final[dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]]

    def __init__(self, config: Config):
        self.config = config
        
        """
        We want the cache file structure to look something like this:
        /root/.cache/huggingface/
        ├── hub/   # regular HF hub cache root for other hub operations
        └── bartowski/
            └── Qwen_Qwen3.5-27B-GGUF/
                ├── Qwen3.5-27B-Q4_K_M.gguf
                └── .cache/huggingface/
        """
        self.hf_cache_model_dir = f"{Globals.HF_CACHE_ROOT_PATH}/{self.config.model_repo}"
        self.hf_cache_model_file = f"{self.hf_cache_model_dir}/{self.config.model_file}" if self.config.model_file is not None else ""
        self.hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)    # pyright: ignore[reportUnknownMemberType]
        self.volumes = {
            Globals.HF_CACHE_ROOT_PATH: self.hf_cache_vol
        }

    @abstractmethod
    def get_base_image(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def image_registry_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError()
    
    @abstractmethod
    def extra_image_setup_commands(self) -> list[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_image_env_vars(self) -> dict[str, str]:
        raise NotImplementedError()
    
    @abstractmethod
    def prewarm_container(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def cmd(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_warmup_endpoint(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_warmup_payload(self) -> dict[str, Any]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_health_check_endpoint(self) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    def get_blocked_admin_paths(self) -> set[str]:
        raise NotImplementedError()
    
    def _download_model_file(self) -> None:
        """
        Downloads a single file from HuggingFace Hub, e.g. one quant from a GGUF repo
        """
        if self.config.model_file is None:
            raise ValueError(f"self.model_file must be specified to download a specific model file, but it was left as None!")
        
        from huggingface_hub import hf_hub_download    # pyright: ignore[reportUnknownVariableType]
        from pathlib import Path

        Path(self.hf_cache_model_dir).mkdir(parents=True, exist_ok=True)

        print(f"Downloading file {self.config.model_repo}/{self.config.model_file} snapshot with requested destination {self.hf_cache_model_dir}...")
        local_path = hf_hub_download(
            repo_id=self.config.model_repo,
            filename=self.config.model_file,
            # revision=MODEL_REVISION,
            local_dir=self.hf_cache_model_dir,
        )
        print(f"Downloading file {self.config.model_repo}/{self.config.model_file} snapshot with requested destination {self.hf_cache_model_dir} done! Actual destination: {local_path}")
        self.hf_cache_vol.commit()

    def _download_repo_snapshot(self) -> None:
        from huggingface_hub import snapshot_download    # pyright: ignore[reportUnknownVariableType]
        from pathlib import Path

        Path(self.hf_cache_model_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading repo {self.config.model_repo} snapshot with requested destination {self.hf_cache_model_dir}...")
        local_path = snapshot_download(
            repo_id=self.config.model_repo,
            # revision=MODEL_REVISION,
            local_dir=self.hf_cache_model_dir,

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
        print(f"Downloading repo {self.config.model_repo} snapshot with requested destination {self.hf_cache_model_dir} done! Actual destination: {local_path}")
        self.hf_cache_vol.commit()
        
    @staticmethod
    def get_hf_env_vars() -> dict[str, Any]:
        return {
            "HF_HOME": Globals.HF_CACHE_ROOT_PATH,
            "HF_HUB_CACHE": f"{Globals.HF_CACHE_ROOT_PATH}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }

    @staticmethod
    def normalize(text: str) -> str:
        return "".join(char for char in text.lower() if char.isalnum())


class AbstractSnapshottableEngine(AbstractEngine, ABC):
    def __init__(self, config: Config):
        if config.n_gpu != 1:
            raise RuntimeError(f"config.n_gpu={config.n_gpu}, but only 1 GPU is allowed for snapshottable containers!")
        
        super().__init__(config)
    
    @abstractmethod
    def get_sleep_endpoint(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_sleep_request_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_wake_endpoint(self) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_wake_request_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError()

