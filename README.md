(One-time setup): Build image and download model files:

```bash
modal run modal_llm_server/modal_inference.py::prewarm_container
```

Then deploy via:

```bash
modal deploy modal_llm_server/modal_inference.py
```
