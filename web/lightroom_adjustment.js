import { app } from "../../../scripts/app.js";

const NODE_TYPES = new Set([
    "MiniMaxH3EasyLightroomImage",
    "MiniMaxH3EasyLightroomVideo",
]);

const HSL_ZONES = [
    ["red", 0], ["orange", 30], ["yellow", 60], ["green", 120],
    ["aqua", 180], ["blue", 210], ["purple", 270], ["magenta", 330],
];

const CONTROL_NAMES = [
    "temperature", "tint", "exposure", "contrast", "highlights", "shadows",
    "whites", "blacks", "texture", "clarity", "dehaze", "vibrance", "saturation",
    ...HSL_ZONES.flatMap(([zone]) => [`${zone}_hue`, `${zone}_saturation`, `${zone}_lightness`]),
];

function widget(node, name) {
    return (node.widgets || []).find((item) => item.name === name);
}

function valuesFor(node) {
    const values = {};
    for (const name of CONTROL_NAMES) values[name] = Number(widget(node, name)?.value || 0);
    return values;
}

function rgbToHsl(red, green, blue) {
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const delta = max - min;
    let hue = 0;
    let saturation = 0;
    const lightness = (max + min) * 0.5;
    if (delta > 1e-6) {
        const denominator = 1 - Math.abs(2 * lightness - 1);
        saturation = denominator > 1e-6 ? delta / denominator : 0;
        if (max === red) hue = ((green - blue) / delta) % 6;
        else if (max === green) hue = (blue - red) / delta + 2;
        else hue = (red - green) / delta + 4;
        hue = ((hue / 6) % 1 + 1) % 1;
    }
    return [hue, saturation, lightness];
}

function hslToRgb(hue, saturation, lightness) {
    saturation = Math.max(0, Math.min(1, saturation));
    lightness = Math.max(0, Math.min(1, lightness));
    const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const hue6 = ((hue % 1) + 1) % 1 * 6;
    const x = chroma * (1 - Math.abs((hue6 % 2) - 1));
    let red = 0; let green = 0; let blue = 0;
    if (hue6 < 1) [red, green, blue] = [chroma, x, 0];
    else if (hue6 < 2) [red, green, blue] = [x, chroma, 0];
    else if (hue6 < 3) [red, green, blue] = [0, chroma, x];
    else if (hue6 < 4) [red, green, blue] = [0, x, chroma];
    else if (hue6 < 5) [red, green, blue] = [x, 0, chroma];
    else [red, green, blue] = [chroma, 0, x];
    const match = lightness - chroma * 0.5;
    return [red + match, green + match, blue + match];
}

function adjustPixel(red, green, blue, controls, localRed, localGreen, localBlue, globalLuminance) {
    const temperature = controls.temperature / 100;
    const tint = controls.tint / 100;
    red += temperature * 0.16 + tint * 0.04;
    green -= tint * 0.10;
    blue -= temperature * 0.16 - tint * 0.04;
    const exposure = Math.pow(2, controls.exposure);
    red *= exposure; green *= exposure; blue *= exposure;
    const contrast = Math.max(-100, Math.min(100, controls.contrast)) / 100;
    red = (red - 0.5) * (1 + contrast) + 0.5;
    green = (green - 0.5) * (1 + contrast) + 0.5;
    blue = (blue - 0.5) * (1 + contrast) + 0.5;
    let luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    const highlightWeight = Math.pow(Math.max(0, Math.min(1, (luminance - 0.45) / 0.55)), 1.5);
    const shadowWeight = Math.pow(Math.max(0, Math.min(1, (0.55 - luminance) / 0.55)), 1.5);
    const tonal = controls.highlights / 100 * highlightWeight * 0.28
        + controls.shadows / 100 * shadowWeight * 0.28
        + controls.whites / 100 * 0.10 + controls.blacks / 100 * 0.10;
    red += tonal; green += tonal; blue += tonal;
    const detailStrength = controls.texture / 100 * 1.25 + controls.clarity / 100 * 0.85;
    red += (red - localRed) * detailStrength;
    green += (green - localGreen) * detailStrength;
    blue += (blue - localBlue) * detailStrength;
    const dehaze = controls.dehaze / 100;
    luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
    const mean = globalLuminance;
    red = (red - mean) * (1 + dehaze * 0.35) + mean + dehaze * 0.04;
    green = (green - mean) * (1 + dehaze * 0.35) + mean + dehaze * 0.04;
    blue = (blue - mean) * (1 + dehaze * 0.35) + mean + dehaze * 0.04;

    let [hue, saturation, lightness] = rgbToHsl(red, green, blue);
    saturation += controls.vibrance / 100 * (1 - saturation) * 0.75;
    saturation *= 1 + controls.saturation / 100;
    for (const [zone, centerDegrees] of HSL_ZONES) {
        let distance = ((hue - centerDegrees / 360 + 0.5) % 1 + 1) % 1 - 0.5;
        const weight = Math.pow(Math.max(0, 1 - Math.abs(distance) / (45 / 360)), 1.5);
        hue += weight * controls[`${zone}_hue`] / 100 * (25 / 360);
        saturation += weight * controls[`${zone}_saturation`] / 100 * 0.50;
        lightness += weight * controls[`${zone}_lightness`] / 100 * 0.25;
    }
    return hslToRgb(hue, saturation, lightness).map((value) => Math.max(0, Math.min(1, value)));
}

function hasAdjustments(controls) {
    return CONTROL_NAMES.some((name) => Math.abs(controls[name]) > 1e-8);
}

function renderPreview(node) {
    const source = node.__h3LightroomSource;
    const canvas = node.__h3LightroomCanvas;
    if (!source || !canvas) return;
    const maxSize = 320;
    const scale = Math.min(1, maxSize / Math.max(source.naturalWidth, source.naturalHeight));
    canvas.width = Math.max(1, Math.round(source.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(source.naturalHeight * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const controls = valuesFor(node);
    if (!hasAdjustments(controls)) return;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const sourcePixels = new Uint8ClampedArray(pixels.data);
    let luminanceSum = 0;
    for (let index = 0; index < sourcePixels.length; index += 4) {
        luminanceSum += (sourcePixels[index] * 0.2126 + sourcePixels[index + 1] * 0.7152 + sourcePixels[index + 2] * 0.0722) / 255;
    }
    const globalLuminance = luminanceSum / Math.max(1, canvas.width * canvas.height);
    for (let index = 0; index < pixels.data.length; index += 4) {
        const pixelIndex = index / 4;
        const x = pixelIndex % canvas.width;
        const y = Math.floor(pixelIndex / canvas.width);
        let localRed = 0; let localGreen = 0; let localBlue = 0; let localCount = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
                const sampleX = Math.max(0, Math.min(canvas.width - 1, x + dx));
                const sampleY = Math.max(0, Math.min(canvas.height - 1, y + dy));
                const sample = (sampleY * canvas.width + sampleX) * 4;
                localRed += sourcePixels[sample] / 255;
                localGreen += sourcePixels[sample + 1] / 255;
                localBlue += sourcePixels[sample + 2] / 255;
                localCount += 1;
            }
        }
        const [red, green, blue] = adjustPixel(
            sourcePixels[index] / 255,
            sourcePixels[index + 1] / 255,
            sourcePixels[index + 2] / 255,
            controls,
            localRed / localCount,
            localGreen / localCount,
            localBlue / localCount,
            globalLuminance,
        );
        pixels.data[index] = Math.round(red * 255);
        pixels.data[index + 1] = Math.round(green * 255);
        pixels.data[index + 2] = Math.round(blue * 255);
    }
    context.putImageData(pixels, 0, 0);
}

function queuePreviewRender(node) {
    if (node.__h3LightroomRenderQueued) return;
    node.__h3LightroomRenderQueued = true;
    requestAnimationFrame(() => {
        node.__h3LightroomRenderQueued = false;
        renderPreview(node);
    });
}

function installPreview(node) {
    if (!node) return;
    if (!node.__h3LightroomInstalled) {
        if (typeof node.addDOMWidget !== "function") return;
        const wrap = document.createElement("div");
        wrap.className = "h3-lightroom-preview-wrap";
        const canvas = document.createElement("canvas");
        canvas.className = "h3-lightroom-preview-canvas";
        canvas.hidden = true;
        wrap.append(canvas);
        const domWidget = node.addDOMWidget("h3_lightroom_preview", "h3_lightroom_preview", wrap, {
            serialize: false,
            hideOnZoom: false,
            canvasOnly: false,
            getMinHeight: () => 150,
            afterResize: () => node.setDirtyCanvas?.(true, true),
        });
        if (!domWidget) {
            wrap.remove();
            return;
        }
        node.__h3LightroomInstalled = true;
        node.__h3LightroomWidget = domWidget;
        node.__h3LightroomCanvas = canvas;
    }
    for (const name of CONTROL_NAMES) {
        const control = widget(node, name);
        if (!control || control.__h3LightroomCallback) continue;
        const original = control.callback;
        control.callback = function lightroomControlChanged() {
            const result = original?.apply(this, arguments);
            queuePreviewRender(node);
            node.setDirtyCanvas?.(true, true);
            return result;
        };
        control.__h3LightroomCallback = true;
    }
}

function setSourcePreview(node, data) {
    if (!data || !node.__h3LightroomCanvas) return;
    const source = new Image();
    source.onload = () => {
        node.__h3LightroomSource = source;
        node.__h3LightroomCanvas.hidden = false;
        queuePreviewRender(node);
        node.setDirtyCanvas?.(true, true);
    };
    source.src = data;
}

app.registerExtension({
    name: "MiniMaxH3Easy.LightroomAdjustment",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData?.name)) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedLightroom() {
            const result = originalCreated?.apply(this, arguments);
            installPreview(this);
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigureLightroom(info) {
            const result = originalConfigure?.apply(this, arguments);
            installPreview(this);
            return result;
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function onExecutedLightroom(output) {
            const result = originalExecuted?.apply(this, arguments);
            installPreview(this);
            setSourcePreview(this, output?.h3_lightroom_preview?.[0]?.data);
            return result;
        };
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onRemovedLightroom() {
            this.__h3LightroomSource = null;
            this.__h3LightroomWidget = null;
            this.__h3LightroomCanvas = null;
            return originalRemoved?.apply(this, arguments);
        };
    },
});

const style = document.createElement("style");
style.textContent = `
  .h3-lightroom-preview-wrap { display:flex; width:100%; min-height:0; align-items:center; justify-content:center; overflow:hidden; background:#111; }
  .h3-lightroom-preview-canvas { display:block; max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; background:#111; }
`;
document.head.append(style);
