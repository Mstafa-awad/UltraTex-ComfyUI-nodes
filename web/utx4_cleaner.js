import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Only modifies UTX 4's node UI; never auto-deletes from a browser load.
app.registerExtension({
    name: "ultratex.v2.cache_cleaner",
    async nodeCreated(node) {
        if (node.comfyClass !== "UTXCacheCleaner" && node.type !== "UTXCacheCleaner") return;

        const readJson = async (response) => {
            const result = await response.json();
            if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
            return result;
        };
        const setStatus = (value) => {
            status.value = value;
            node.setDirtyCanvas?.(true, true);
        };
        const status = node.addWidget("text", "Cache status", "Click Scan Size", () => {}, { serialize: false });
        status.options = { ...(status.options || {}), serialize: false };

        node.addWidget("button", "Scan Size", "Scan Size", async () => {
            setStatus("Scanning...");
            try {
                const result = await readJson(await api.fetchApi("/ultratex/v2/cleaner/scan"));
                setStatus(`${result.folder_count} folders / ${result.total_gib.toFixed(3)} GiB`);
                const details = result.folders.map(f => `${f.name}: ${(f.bytes / 1048576).toFixed(2)} MiB${f.deletable ? "" : " (not safe to delete)"}`).join("\n");
                window.alert(`UltraTex temporary cache\n${result.total_gib.toFixed(3)} GiB in ${result.folder_count} folders\n\n${details || "No temporary caches found"}\n\nProtected: output/ultratex (all GLBs), output/3d and other folders`);
            } catch (error) {
                setStatus(`Scan failed: ${error.message}`);
                window.alert(`UTX 4 scan failed: ${error.message}`);
            }
        });

        node.addWidget("button", "Delete Now", "Delete Now", async () => {
            if (!window.confirm("DELETE UltraTex temporary caches?\n\nOnly output/ultratex_views_* and output/.ultratex_cache will be removed.\n\nThis CANNOT be undone. Make sure no UltraTex job is running or needs cached intermediates.\n\noutput/ultratex (saved GLBs) and output/3d stay untouched.")) return;
            setStatus("Deleting temporary caches...");
            try {
                const response = await api.fetchApi("/ultratex/v2/cleaner/delete", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ confirm: "DELETE_ULTRATEX_TEMP" }),
                });
                const result = await readJson(response);
                setStatus(`Reclaimed ${result.reclaimed_mib.toFixed(2)} MiB; ${result.after.folder_count} remain`);
                const extra = [
                    ...(result.refused_folders || []).map(x => `Skipped unsafe: ${x}`),
                    ...(result.errors || []).map(x => `Error: ${x}`),
                ].join("\n");
                window.alert(`UltraTex cleanup finished.\n\nRemoved: ${result.removed_folders.join(", ") || "none"}\nReclaimed ~${result.reclaimed_mib.toFixed(2)} MiB\nRemaining: ${result.after.total_mib.toFixed(2)} MiB\n\noutput/ultratex is untouched.${extra ? "\n\n" + extra : ""}`);
            } catch (error) {
                setStatus(`Delete failed: ${error.message}`);
                window.alert(`UTX 4 delete failed: ${error.message}`);
            }
        });
        node.setSize?.([Math.max(node.size?.[0] || 0, 330), Math.max(node.size?.[1] || 0, 200)]);
    },
});
