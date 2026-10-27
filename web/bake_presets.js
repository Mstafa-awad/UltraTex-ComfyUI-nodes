import { app } from "../../scripts/app.js";

const aliases = {
    fast_detail: "Quality", off_direct: "Fast", detail_consistent: "Quality",
    sharp_source: "Quality", sharp_coherent: "Quality", sharp_anchor: "Quality",
    sharp_anchor_v2: "Quality", sharp_anchor_v3: "Quality", sharp_anchor_v4: "Quality",
    "Sharp anchor": "Quality", "Quality + edge repair": "Quality",
};

export function migrateBakeNode(info) {
    if (info?.type === "UTXIntelligentViewEnhancer") {
        const connected = info.inputs?.some(input => input.name === "upscale_model" && input.link != null);
        const replacement = connected ? "Comfy upscale model" : "None";
        if (Array.isArray(info.widgets_values) && info.widgets_values[5] === "VOSR2") {
            info.widgets_values[5] = replacement;
            if (!connected && info.widgets_values[6] === "Native model scale") info.widgets_values[6] = "Original";
        }
        if (info.widgets_values_named?.upscaler === "VOSR2") {
            info.widgets_values_named.upscaler = replacement;
            if (!connected && info.widgets_values_named.target === "Native model scale") info.widgets_values_named.target = "Original";
        }
        return info;
    }
    if (info?.type !== "UTXFastSurfaceTextureBaker") return info;
    const values = info.widgets_values;
    if (Array.isArray(values)) {
        // The retired completion widget was between texture size and mode.
        if (values[1] === "surface_fill" || values[1] === "off") values.splice(1, 1);
        values[1] = aliases[values[1]] ?? values[1];
    } else if (values && typeof values === "object") {
        values.upscale_mode = aliases[values.upscale_mode] ?? values.upscale_mode;
        delete values.completion;
    }
    if (info.widgets_values_named) {
        const named = info.widgets_values_named;
        named.upscale_mode = aliases[named.upscale_mode] ?? named.upscale_mode;
        delete named.completion;
    }
    return info;
}

app.registerExtension({
    name: "ultratex.supported_bake_presets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["UTXFastSurfaceTextureBaker", "UTXIntelligentViewEnhancer"].includes(nodeData.name)) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            return configure.call(this, migrateBakeNode(info));
        };
    },
});
