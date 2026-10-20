"""UltraTex Modular. Importing the package never loads models or GPU backends."""

__version__ = "2.2.0rc4"


async def comfy_entrypoint():
    from .core.cpu import configure_cpu
    configure_cpu()
    from .startup import apply_native_cap
    apply_native_cap()
    from .utx_cache_cleaner import register_routes
    register_routes()
    from .extension import UltraTexExtension
    return UltraTexExtension()


# JavaScript adds the Scan Size / Delete Now buttons to UTX 4.
WEB_DIRECTORY = "./web"
