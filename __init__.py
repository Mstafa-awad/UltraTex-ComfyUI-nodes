"""UltraTex Modular. Importing the package never loads models or GPU backends."""

__version__ = "2.3.8rc1"


async def comfy_entrypoint():
    try:
        # Cache folders of ComfyUI sessions that ended (crash, closed window) and old flat-cache files.
        import folder_paths
        from .core.runtime import remove_stale_sessions
        remove_stale_sessions(folder_paths.get_output_directory())
    except Exception as error:  # cleanup must never stop the node pack from loading
        print(f"[UltraTex] stale cache cleanup skipped: {error}", flush=True)
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
