"""UltraTex Modular. Importing the package never loads models or GPU backends."""

__version__ = "0.2.6.1"


async def comfy_entrypoint():
    from .extension import UltraTexExtension
    return UltraTexExtension()
