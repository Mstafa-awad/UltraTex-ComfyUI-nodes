"""UTX 4: explicit, name-scoped cleanup of temporary UltraTex output folders.

Only direct child directories named .ultratex_cache or ultratex_views_<id>
under ComfyUI's official output directory may be deleted. The output/ultratex
folder (finished GLBs/textures) and other output entries are NEVER candidates.
This module intentionally does not import CUDA, Torch, or UltraTex generation code.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any

from comfy_api.latest import io


_VIEW_BUNDLE = re.compile(r"^ultratex_views_[A-Za-z0-9_-]{1,100}$", flags=re.ASCII)
_CONFIRM = "DELETE_ULTRATEX_TEMP"
_ROUTES_REGISTERED = False


def _is_candidate_name(name: str) -> bool:
    return name == ".ultratex_cache" or bool(_VIEW_BUNDLE.fullmatch(name))


def _is_reparse(path: Path) -> bool:
    """Reject symlinks, Windows junctions, and other directory reparse points."""
    try:
        info = path.lstat()
        return bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT) if hasattr(info, "st_file_attributes") else path.is_symlink()
    except OSError:
        return True


def _official_output_dir() -> Path:
    import folder_paths
    path = Path(folder_paths.get_output_directory()).resolve(strict=True)
    if not path.is_dir() or path == Path(path.anchor):
        raise ValueError("ComfyUI output path is not a safe directory")
    return path


def _safe_candidates(output_dir: Path) -> list[Path]:
    """Never follow nested paths or user-supplied root paths."""
    if not output_dir.is_dir():
        raise ValueError("ComfyUI output path does not exist")
    candidates = []
    with os.scandir(output_dir) as entries:
        for entry in entries:
            if not _is_candidate_name(entry.name):
                continue
            path = output_dir / entry.name
            if path.parent != output_dir or _is_reparse(path):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    candidates.append(path)
            except OSError:
                continue
    return sorted(candidates, key=lambda p: p.name)


def _tree_bytes(directory: Path) -> tuple[int, int, int, bool, str]:
    """Return logical file bytes, files, directories, has_reparse, any error.

    We count only regular files, never follow a symlink or junction, and
    report any scan errors rather than silently claiming a complete size.
    """
    total = files = 0
    dirs = 1
    unsafe = False
    errors = []
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    item = Path(entry.path)
                    if _is_reparse(item):
                        unsafe = True
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs += 1
                            stack.append(item)
                        elif entry.is_file(follow_symlinks=False):
                            files += 1
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError as exc:
                        errors.append(f"{item.name}: {exc}")
        except OSError as exc:
            errors.append(f"{current.name}: {exc}")
    return total, files, dirs, unsafe, "; ".join(errors[:3])


def scan_cache(output_dir: Path | None = None) -> dict[str, Any]:
    """Inventory deletable items. Not destructive; independent of node graph."""
    root = output_dir if output_dir is not None else _official_output_dir()
    root = Path(root).resolve(strict=True)
    folders = []
    for candidate in _safe_candidates(root):
        bytes_used, file_count, dir_count, unsafe, error = _tree_bytes(candidate)
        folders.append({
            "name": candidate.name,
            "bytes": bytes_used,
            "files": file_count,
            "subdirectories_including_root": dir_count,
            "deletable": not unsafe and not error,
            "problem": ("Contains a link/junction; refused for safety" if unsafe else error),
        })
    total = sum(f["bytes"] for f in folders)
    return {
        "output_directory": str(root),
        "folders": folders,
        "folder_count": len(folders),
        "file_count": sum(f["files"] for f in folders),
        "total_bytes": total,
        "total_mib": round(total / (1024 ** 2), 2),
        "total_gib": round(total / (1024 ** 3), 3),
        "protected": ["ultratex", "3d", "every nonmatching output folder and file"],
    }


def delete_cache(output_dir: Path | None = None) -> dict[str, Any]:
    """Delete only inventoried approved child folders; never output/ultratex."""
    root = output_dir if output_dir is not None else _official_output_dir()
    root = Path(root).resolve(strict=True)
    before = scan_cache(root)
    removed, refused, errors = [], [], []
    for details in before["folders"]:
        name = details["name"]
        child = root / name
        if not _is_candidate_name(name) or child.parent != root:
            refused.append(name)
            continue
        # Re-check immediately before deletion; refuse a linked or mutated tree.
        if _is_reparse(child) or not child.is_dir():
            refused.append(name)
            continue
        _, _, _, unsafe, error = _tree_bytes(child)
        if unsafe or error:
            refused.append(name)
            continue
        try:
            shutil.rmtree(child)
            removed.append(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    after = scan_cache(root)
    if removed and output_dir is None:
        _forget_cached_node_outputs()
    return {
        "before": before,
        "after": after,
        "removed_folders": removed,
        "refused_folders": refused,
        "errors": errors,
        "reclaimed_logical_bytes": max(0, before["total_bytes"] - after["total_bytes"]),
        "reclaimed_mib": round(max(0, before["total_bytes"] - after["total_bytes"]) / (1024 ** 2), 2),
        "success": not refused and not errors,
    }


def _forget_cached_node_outputs() -> bool:
    """Make ComfyUI drop its cached node outputs once the current run ends (models stay loaded).

    Those outputs can point at the arrays just deleted; without this a later run that changes only a downstream
    setting would reuse them and fail. With it, the next run simply recomputes."""
    try:
        from server import PromptServer
        queue = PromptServer.instance.prompt_queue
        with queue.mutex:
            queue.flags["unload_models"] = False
            queue.flags["free_memory"] = True
            queue.not_empty.notify()
        return True
    except Exception:
        return False


def _format_summary(result: dict[str, Any]) -> str:
    return f'{result["folder_count"]} temporary folder(s) · {result["file_count"]:,} file(s) · {result["total_gib"]:.3f} GiB ({result["total_mib"]:.2f} MiB)'


def register_routes() -> None:
    """The frontend's two explicit buttons use same-origin authenticated Comfy API."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/ultratex/v2/cleaner/scan")
    async def _scan(request):
        try:
            result = await asyncio.to_thread(scan_cache)
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/ultratex/v2/cleaner/delete")
    async def _delete(request):
        try:
            data = await request.json()
            if not isinstance(data, dict) or data.get("confirm") != _CONFIRM:
                return web.json_response({"error": "Explicit confirmation missing"}, status=400)
            result = await asyncio.to_thread(delete_cache)
            return web.json_response(result)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    _ROUTES_REGISTERED = True


class UTXCacheCleaner(io.ComfyNode):
    """V3 node: cleanup trigger + pass-through of arbitrary connected value."""

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("utx4_passthrough")
        return io.Schema(
            node_id="UTXCacheCleaner",
            display_name="UTX 4. Cache Cleaner",
            category="UltraTex Modular",
            inputs=[
                io.MatchType.Input("passthrough", template=template, optional=True, tooltip="Optional input forwarded unchanged to output. Connect at end or leave empty at start."),
                io.Combo.Input("mode", options=["scan_only", "delete_when_executed"], default="scan_only",
                               tooltip="Choose delete_when_executed to clean after upstream finishes. Delete Now button also works independently."),
                io.Boolean.Input("confirm_delete_on_run", default=False,
                                 tooltip="Extra safety confirmation required for automatic workflow deletion."),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="passthrough"),
                io.String.Output(display_name="cache_size"),
                io.String.Output(display_name="cleanup_report"),
            ],
            is_output_node=True,
            description="Shows UltraTex temporary cache size; deletes only output/.ultratex_cache and output/ultratex_views_* with explicit confirmation, then lets ComfyUI forget the deleted intermediates so the next run recomputes them. Never deletes output/ultratex or saved GLBs. Without it the cache is still removed when ComfyUI exits.",
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Fresh status for every queued run instead of returning a cached report.
        return float("nan")

    @classmethod
    def execute(cls, mode="scan_only", confirm_delete_on_run=False, passthrough=None):
        from comfy_api.latest import io
        try:
            if mode == "delete_when_executed" and confirm_delete_on_run:
                result = delete_cache()
                report = (f'Reclaimed {result["reclaimed_mib"]:.2f} MiB; removed: '
                          f'{", ".join(result["removed_folders"]) or "none"}; '
                          f'refused: {", ".join(result["refused_folders"]) or "none"}; '
                          f'errors: {", ".join(result["errors"]) or "none"}. '
                          'output/ultratex and GLBs preserved.')
                current = result["after"]
            else:
                current = scan_cache()
                report = ('SCAN ONLY — no deletion performed. ' +
                          ('To auto-delete, enable confirm_delete_on_run.' if mode == "delete_when_executed" else
                           'Click Delete Now in this node to remove temporary folders.') +
                          ' output/ultratex remains protected.')
            summary = _format_summary(current)
            print(f'[UTX 4 Cleaner] {summary} | {report}', flush=True)
            return io.NodeOutput(passthrough, summary, report)
        except Exception as exc:
            message = f'Cache scan/cleanup failed (saved files remain protected): {exc}'
            print(f'[UTX 4 Cleaner] {message}', flush=True)
            return io.NodeOutput(passthrough, 'Scan failed', message)
