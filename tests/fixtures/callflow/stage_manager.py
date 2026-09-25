import asyncio


async def launch_stage_ov():
    await open_stage()
    await wait_for_stage_load()
    await preload_prefabs()


async def open_stage():
    asyncio.create_task(_load_async())
    _init_ui()


async def _load_async():
    await _fetch_assets()


async def _fetch_assets():
    pass


def _init_ui():
    pass


async def wait_for_stage_load():
    await asyncio.gather(_wait_ready())


async def _wait_ready():
    pass


async def preload_prefabs():
    pass