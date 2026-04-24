1. Set up config (and change active config) in modal_llm_server/config.py

2. (One-time setup per model): Build image and download model files:

```bash
modal run -m modal_llm_server.modal_server::prewarm_container
```

3. Then deploy via:

```bash
modal deploy -m modal_llm_server.modal_server
```

NOTE: Snapshottable server in modal_llm_server/modal_snapshot_server.py is still not working due to issues with VLLM and SGLang images interacting with B200 GPUs and Modal snapshot recovery mechanism.
