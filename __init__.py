"""UltraTex Modular. Importing the package never loads models or GPU backends."""

__version__ = "2.0.0"


async def comfy_entrypoint():
    from .startup import apply_native_cap
    apply_native_cap()
    from .extension import UltraTexExtension
    return UltraTexExtension()
