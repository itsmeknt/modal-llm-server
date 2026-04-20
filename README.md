1. Set up config (and change active config) in modal_llm_server/config.py

2. (One-time setup per model): Build image and download model files:

```bash
modal run modal_llm_server/modal_inference.py::prewarm_container
```

3. Then deploy via:

```bash
modal deploy modal_llm_server/modal_inference.py
```

NOTE: Snapshottable instances in modal_llm_server/modal_snapshot_inference.py is still not working due to issues with VLLM and SGLang images interacting with B200 GPUs and Modal snapshot recovery mechanism.
