# pyright: reportUnknownMemberType=false

import httpx
import modal
from modal_llm_server.base_modal_server import CONFIG, ENGINE, IMAGE, ServeBase, app_name
from modal_llm_server.config import Globals
from modal_llm_server.engines.abstract_engine import AbstractSnapshottableEngine


if not isinstance(ENGINE, AbstractSnapshottableEngine):
    raise RuntimeError(f"Engine {ENGINE.__class__.__name__} is not an snapshottable engine! Cannot deploy snapshottable Modal server.")

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
    
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=ENGINE.config.max_num_seqs)
class Serve(ServeBase):
    async def _sleep_server(self) -> None:
        r = await self.client.post(
            ENGINE.get_sleep_endpoint(),
            **ENGINE.get_sleep_request_kwargs(),
            timeout=300.0,
        )
        _ = r.raise_for_status()

    async def _wake_server(self) -> None:
        r = await self.client.post(
            ENGINE.get_wake_endpoint(),
            **ENGINE.get_wake_request_kwargs(),
            timeout=300.0,
        )
        _ = r.raise_for_status()
        await self.wait_ready()

    @modal.enter(snap=True)
    async def start_engine_and_snapshot(self):
        await self.start_engine_base()
        await self._sleep_server()

        # do not carry a live client socket pool into restore        
        await self.client.aclose()

    @modal.enter(snap=False)
    async def wake_after_restore(self):
        self.upstream: str = f"http://127.0.0.1:{ENGINE.config.port}"
        self.client: httpx.AsyncClient = self.new_client()
        await self._wake_server()

    @modal.exit()
    async def stop_engine(self):
        await self.stop_engine_base()
