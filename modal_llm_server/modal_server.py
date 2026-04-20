# pyright: reportUnknownMemberType=false

# modal_llm_server/modal_inference.py
import modal
from modal_llm_server.base_modal_server import CONFIG, ENGINE, IMAGE, ServeBase, app_name
from modal_llm_server.config import Globals

app = modal.App(app_name(CONFIG))

@app.function(image=IMAGE, volumes=ENGINE.volumes, timeout=Globals.PREWARM_TIMEOUT_S)
def prewarm_container():
    ENGINE.prewarm_container()

@app.cls(
    image=IMAGE,
    gpu=f"{ENGINE.config.gpu_type}:{ENGINE.config.n_gpu}",
    scaledown_window=Globals.SCALEDOWN_S,
    timeout=Globals.TIMEOUT_S,
    volumes=ENGINE.volumes,
)
@modal.concurrent(max_inputs=ENGINE.config.max_num_seqs)
class Serve(ServeBase):
    @modal.enter()
    async def start_engine(self):
        await self.start_engine_base()

    @modal.exit()
    async def stop_engine(self):
        await self.stop_engine_base()

