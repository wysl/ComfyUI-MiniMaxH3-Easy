import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "MiniMaxH3EasyMediaLoader";
const MANIFEST_WIDGET = "media_manifest";
const EMPTY_MANIFEST = { version: 1, images: [], audios: [], videos: [] };
const DEFAULT_SCALE = 0.5;
const DEFAULT_EDGE_LENGTH = 1024;
const FIXED_SCALE_METHOD = "lanczos";
const INPUT_PICKER_PAGE_SIZE = 48;
const SCALE_METHODS = new Set(["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]);
const RESIZE_MODES = new Set(["不缩放", "原图", "倍率", "长边", "短边", "none", "original", "disabled", "long edge", "long_edge", "short edge", "short_edge"]);
const KIND_CONFIG = {
    images: {
        label: "图片",
        singular: "图片",
        accept: ".png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff,image/*",
        extensions: new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]),
    },
    audios: {
        label: "音频",
        singular: "音频",
        accept: ".wav,.mp3,.flac,.ogg,.m4a,.aac,audio/*",
        extensions: new Set(["wav", "mp3", "flac", "ogg", "m4a", "aac"]),
    },
    videos: {
        label: "视频",
        singular: "视频",
        accept: ".mp4,.mov,.mkv,.webm,.avi,.m4v,video/*",
        extensions: new Set(["mp4", "mov", "mkv", "webm", "avi", "m4v"]),
    },
};

function cloneManifest(manifest) {
    return {
        version: 1,
        images: [...manifest.images],
        audios: [...manifest.audios],
        videos: [...manifest.videos],
    };
}

function parseManifest(value) {
    try {
        const parsed = typeof value === "string" ? JSON.parse(value || "{}") : value;
        const manifest = cloneManifest(EMPTY_MANIFEST);
        for (const kind of Object.keys(KIND_CONFIG)) {
            if (!Array.isArray(parsed?.[kind])) continue;
            manifest[kind] = parsed[kind]
                .map((item) => String(item || "").trim())
                .filter(Boolean);
        }
        return manifest;
    } catch {
        return cloneManifest(EMPTY_MANIFEST);
    }
}

function widgetByName(node, name) {
    return node?.widgets?.find((widget) => widget.name === name);
}

function hideManifestWidget(widget) {
    if (!widget || widget.__h3MediaLoaderHidden) return;
    widget.__h3MediaLoaderHidden = true;
    widget.__h3MediaLoaderOriginalType = widget.type;
    widget.__h3MediaLoaderOriginalComputeSize = widget.computeSize;
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.options ||= {};
    widget.options.hidden = true;
    widget.options.canvasOnly = true;
    if (widget.inputEl) widget.inputEl.style.display = "none";
}

function commitManifest(node, manifest) {
    const widget = widgetByName(node, MANIFEST_WIDGET);
    if (!widget) return;
    const value = JSON.stringify(cloneManifest(manifest));
    if (widget.value === value) return;
    node.graph?.beforeChange?.();
    widget.value = value;
    widget.callback?.(value);
    node.graph?.afterChange?.();
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function uploadedPath(upload) {
    const name = String(upload?.name || "").trim();
    const subfolder = String(upload?.subfolder || "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
    return subfolder ? `${subfolder}/${name}` : name;
}

async function uploadFile(file, kind) {
    const form = new FormData();
    form.append("image", file, file.name);
    form.append("type", "input");
    form.append("overwrite", "false");
    form.append("subfolder", `minimax_h3_easy/media_loader/${kind}`);
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const path = uploadedPath(await response.json());
    if (!path) throw new Error("Upload response did not contain a file name");
    return path;
}

function extensionOf(name) {
    const match = String(name || "").toLowerCase().match(/\.([^.]+)$/);
    return match?.[1] || "";
}

function displayName(path) {
    const normalized = String(path || "").replaceAll("\\", "/");
    return normalized.slice(normalized.lastIndexOf("/") + 1);
}

function imagePreviewUrl(path) {
    const normalized = String(path || "").replaceAll("\\", "/").replace(/^\/+/, "");
    const params = new URLSearchParams({ filename: normalized });
    return api.apiURL(`/minimax_h3_easy/input-preview?${params.toString()}`);
}

function iconForKind(kind) {
    if (kind === "audios") return "AUDIO";
    if (kind === "videos") return "VIDEO";
    return "IMAGE";
}

function createIconButton(label, title, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `h3-media-icon-button ${className}`.trim();
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);
    return button;
}

function normalizedResizeMode(value) {
    const mode = String(value || "倍率").trim().toLowerCase();
    if (["不缩放", "原图", "none", "original", "disabled"].includes(mode)) return "none";
    if (["长边", "long edge", "long_edge"].includes(mode)) return "long_edge";
    if (["短边", "short edge", "short_edge"].includes(mode)) return "short_edge";
    return "scale";
}

function isNumericWidgetValue(value) {
    return value !== null && value !== "" && Number.isFinite(Number(value));
}

function isScaleMethodValue(value) {
    return SCALE_METHODS.has(String(value || "").trim().toLowerCase());
}

function isResizeModeValue(value) {
    return RESIZE_MODES.has(String(value || "").trim().toLowerCase());
}

function setWidgetValue(widget, value) {
    if (!widget || widget.value === value) return false;
    widget.value = value;
    if (widget._state) widget._state.value = value;
    return true;
}

function repairLegacyResizeWidgetValues(node) {
    const scaleWidget = widgetByName(node, "image_scale");
    const methodWidget = widgetByName(node, "scale_method");
    const resizeModeWidget = widgetByName(node, "image_resize_mode");
    if (!scaleWidget || !methodWidget || !resizeModeWidget) return false;

    // Older workflows serialized these three widgets as scale, method, mode.
    // The current schema is mode, scale, method, so identify that shape by type
    // and restore the values before the controls are rendered or submitted.
    if (
        isScaleMethodValue(scaleWidget.value)
        && isResizeModeValue(methodWidget.value)
        && isNumericWidgetValue(resizeModeWidget.value)
    ) {
        const legacyScale = Number(resizeModeWidget.value);
        const legacyMode = String(methodWidget.value);
        setWidgetValue(resizeModeWidget, legacyMode);
        setWidgetValue(scaleWidget, legacyScale);
        setWidgetValue(methodWidget, FIXED_SCALE_METHOD);
        node.graph?.change?.();
        return true;
    }
    return false;
}

function applyResizeModeDefaults(node, mode) {
    const previousMode = node.__h3MediaLoaderResizeMode;
    let changed = false;
    const scaleWidget = widgetByName(node, "image_scale");
    const methodWidget = widgetByName(node, "scale_method");
    const edgeLengthWidget = widgetByName(node, "image_edge_length");

    if (previousMode !== undefined && previousMode !== mode) {
        if (mode === "scale") changed = setWidgetValue(scaleWidget, DEFAULT_SCALE) || changed;
        if (mode === "long_edge" || mode === "short_edge") {
            changed = setWidgetValue(edgeLengthWidget, DEFAULT_EDGE_LENGTH) || changed;
        }
    }
    changed = setWidgetValue(methodWidget, FIXED_SCALE_METHOD) || changed;
    node.__h3MediaLoaderResizeMode = mode;
    if (changed) node.graph?.change?.();
}

function setResizeWidgetVisibility(widget, visible) {
    if (!widget) return;
    if (!widget.__h3MediaLoaderOriginalType) {
        widget.__h3MediaLoaderOriginalType = widget.type;
        widget.__h3MediaLoaderOriginalComputeSize = widget.computeSize;
    }
    widget.hidden = !visible;
    widget.type = visible ? widget.__h3MediaLoaderOriginalType : "hidden";
    widget.computeSize = visible
        ? widget.__h3MediaLoaderOriginalComputeSize
        : () => [0, -4];
    widget.options ||= {};
    widget.options.hidden = !visible;
    if (widget.inputEl) widget.inputEl.style.display = visible ? "" : "none";
}

function refreshResizeControls(node) {
    const scaleWidget = widgetByName(node, "image_scale");
    const methodWidget = widgetByName(node, "scale_method");
    const resizeModeWidget = widgetByName(node, "image_resize_mode");
    const edgeLengthWidget = widgetByName(node, "image_edge_length");
    const divisibleByWidget = widgetByName(node, "image_divisible_by");
    const mode = normalizedResizeMode(resizeModeWidget?.value);
    const usesScale = mode === "scale";
    const usesEdge = mode === "long_edge" || mode === "short_edge";

    applyResizeModeDefaults(node, mode);

    setResizeWidgetVisibility(scaleWidget, usesScale);
    setResizeWidgetVisibility(edgeLengthWidget, usesEdge);
    setResizeWidgetVisibility(methodWidget, mode !== "none");
    setResizeWidgetVisibility(divisibleByWidget, mode !== "none");

    if (resizeModeWidget) resizeModeWidget.label = "图片缩放模式（唯一生效规则）";
    if (scaleWidget) {
        scaleWidget.label = usesScale ? "自定义缩放倍率（当前生效）" : "自定义缩放倍率（已忽略）";
    }
    if (edgeLengthWidget) {
        edgeLengthWidget.label = usesEdge ? "目标边长/像素（当前生效）" : "目标边长/像素（已忽略）";
    }
    if (methodWidget) {
        methodWidget.label = mode === "none" ? "缩放算法（已忽略）" : "图片缩放算法";
    }
    if (divisibleByWidget) {
        divisibleByWidget.label = mode === "none"
            ? "尺寸因数（已忽略）"
            : "尺寸因数（宽高可整除，1=关闭）";
    }

    const status = node.__h3MediaLoaderResizeStatus;
    if (status) {
        const divisibleBy = Math.max(1, Number.parseInt(divisibleByWidget?.value ?? 1, 10) || 1);
        const factorNote = divisibleBy > 1 ? `；宽高对齐因数 ${divisibleBy}` : "；尺寸因数关闭";
        if (mode === "none") {
            status.textContent = "当前：保持原图尺寸（其他缩放参数均不生效）";
        } else if (mode === "long_edge") {
            status.textContent = `当前：长边缩放至 ${edgeLengthWidget?.value ?? 1024}px（倍率不生效）${factorNote}`;
        } else if (mode === "short_edge") {
            status.textContent = `当前：短边缩放至 ${edgeLengthWidget?.value ?? 1024}px（倍率不生效）${factorNote}`;
        } else {
            status.textContent = `当前：按自定义倍率 ×${Number(scaleWidget?.value ?? DEFAULT_SCALE).toFixed(2)} 缩放（边长不生效）${factorNote}`;
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function watchResizeControls(node) {
    for (const name of ["image_resize_mode", "image_scale", "image_edge_length", "image_divisible_by", "scale_method"]) {
        const widget = widgetByName(node, name);
        if (!widget || widget.__h3MediaLoaderWatched) continue;
        widget.__h3MediaLoaderWatched = true;
        const originalCallback = widget.callback;
        widget.callback = function h3MediaLoaderResizeChanged() {
            const result = originalCallback?.apply(this, arguments);
            refreshResizeControls(node);
            return result;
        };
    }
}

function setupMediaLoader(node) {
    const manifestWidget = widgetByName(node, MANIFEST_WIDGET);
    hideManifestWidget(manifestWidget);
    repairLegacyResizeWidgetValues(node);
    const scaleWidget = widgetByName(node, "image_scale");
    const methodWidget = widgetByName(node, "scale_method");
    const resizeModeWidget = widgetByName(node, "image_resize_mode");
    const edgeLengthWidget = widgetByName(node, "image_edge_length");
    watchResizeControls(node);
    refreshResizeControls(node);
    if (node.__h3MediaLoaderWidget) {
        node.__h3MediaLoaderRender?.();
        return;
    }

    node.resizable = true;
    node.min_size = [300, 280];

    const root = document.createElement("div");
    root.className = "h3-media-loader";
    const resizeStatus = document.createElement("div");
    resizeStatus.className = "h3-media-resize-status";
    const tabs = document.createElement("div");
    tabs.className = "h3-media-tabs";
    const toolbar = document.createElement("div");
    toolbar.className = "h3-media-toolbar";
    const addButton = createIconButton("+", "添加媒体", "h3-media-add");
    const inputButton = createIconButton("⇩", "从 input 目录选择媒体", "h3-media-input");
    const clearButton = createIconButton("×", "清空当前分类", "h3-media-clear");
    const status = document.createElement("div");
    status.className = "h3-media-status";
    toolbar.append(addButton, inputButton, clearButton, status);
    const content = document.createElement("div");
    content.className = "h3-media-content";
    root.append(resizeStatus, tabs, toolbar, content);

    const state = {
        activeKind: "images",
        uploading: false,
        drag: null,
        inputPickerOpen: false,
        inputFiles: [],
        inputPage: 0,
        inputSelection: new Set(),
    };

    const currentManifest = () => parseManifest(widgetByName(node, MANIFEST_WIDGET)?.value);
    const setStatus = (message, error = false) => {
        status.textContent = message;
        status.classList.toggle("is-error", error);
    };

    const reorder = (kind, from, to) => {
        if (kind !== state.activeKind || from === to) return;
        const manifest = currentManifest();
        const items = manifest[kind];
        if (from < 0 || to < 0 || from >= items.length || to >= items.length) return;
        const [item] = items.splice(from, 1);
        items.splice(to, 0, item);
        commitManifest(node, manifest);
        render();
    };

    const removeAt = (kind, index) => {
        const manifest = currentManifest();
        manifest[kind].splice(index, 1);
        commitManifest(node, manifest);
        render();
    };

    const addLocalFiles = async (selectedFiles, kind = state.activeKind) => {
        const config = KIND_CONFIG[kind];
        const files = Array.from(selectedFiles || []).filter((file) =>
            config.extensions.has(extensionOf(file.name))
        );
        if (!files.length) {
            if (selectedFiles?.length) setStatus(`未找到可用${config.singular}`, true);
            return;
        }

        state.uploading = true;
        render();
        setStatus(`正在添加 0/${files.length}`);
        const uploaded = [];
        let uploadError = null;
        try {
            for (let index = 0; index < files.length; index += 1) {
                uploaded.push(await uploadFile(files[index], kind));
                setStatus(`正在添加 ${index + 1}/${files.length}`);
            }
        } catch (error) {
            uploadError = error;
        } finally {
            if (uploaded.length) {
                const manifest = currentManifest();
                manifest[kind].push(...uploaded);
                commitManifest(node, manifest);
            }
            if (uploadError) {
                setStatus(
                    `已添加 ${uploaded.length}/${files.length}: ${uploadError?.message || uploadError}`,
                    true,
                );
            } else {
                setStatus(`已添加 ${uploaded.length} 项`);
            }
            state.uploading = false;
            render();
        }
    };

    const toggleInputSelection = (path) => {
        if (state.inputSelection.has(path)) state.inputSelection.delete(path);
        else state.inputSelection.add(path);
        render();
    };

    const addSelectedInputFiles = () => {
        const selected = [...state.inputSelection];
        if (!selected.length) {
            setStatus("请先选择 input 媒体", true);
            return;
        }
        const manifest = currentManifest();
        const existing = new Set(manifest[state.activeKind]);
        manifest[state.activeKind].push(...selected.filter((path) => !existing.has(path)));
        commitManifest(node, manifest);
        state.inputPickerOpen = false;
        state.inputFiles = [];
        state.inputSelection.clear();
        setStatus(`已加入 ${selected.length} 项 input 媒体`);
        render();
    };

    const renderInputPicker = () => {
        const picker = document.createElement("div");
        picker.className = "h3-media-input-picker";
        const header = document.createElement("div");
        header.className = "h3-media-input-picker-header";
        const title = document.createElement("span");
        title.textContent = `input 目录 · ${KIND_CONFIG[state.activeKind].label}`;
        const actions = document.createElement("div");
        actions.className = "h3-media-input-picker-actions";
        const pageCount = Math.max(1, Math.ceil(state.inputFiles.length / INPUT_PICKER_PAGE_SIZE));
        state.inputPage = Math.min(Math.max(0, state.inputPage), pageCount - 1);
        const pageLabel = document.createElement("span");
        pageLabel.className = "h3-media-input-page";
        pageLabel.textContent = `${state.inputPage + 1}/${pageCount}`;
        const previousPageButton = createIconButton(
            "‹",
            "上一页 input 媒体",
            "h3-media-input-page-previous",
        );
        previousPageButton.disabled = state.inputPage === 0;
        previousPageButton.addEventListener("click", () => {
            state.inputPage -= 1;
            render();
        });
        const nextPageButton = createIconButton(
            "›",
            "下一页 input 媒体",
            "h3-media-input-page-next",
        );
        nextPageButton.disabled = state.inputPage >= pageCount - 1;
        nextPageButton.addEventListener("click", () => {
            state.inputPage += 1;
            render();
        });
        const addSelectedButton = createIconButton(
            "✓",
            "加入选中的 input 媒体",
            "h3-media-input-add",
        );
        addSelectedButton.disabled = state.inputSelection.size === 0;
        addSelectedButton.addEventListener("click", addSelectedInputFiles);
        const closeButton = createIconButton("×", "关闭 input 媒体选择", "h3-media-input-close");
        closeButton.addEventListener("click", () => {
            state.inputPickerOpen = false;
            state.inputFiles = [];
            state.inputSelection.clear();
            render();
        });
        actions.append(previousPageButton, pageLabel, nextPageButton, addSelectedButton, closeButton);
        header.append(title, actions);
        picker.append(header);

        if (!state.inputFiles.length) {
            const empty = document.createElement("div");
            empty.className = "h3-media-input-empty";
            empty.textContent = "input 目录没有匹配的媒体";
            picker.append(empty);
            content.append(picker);
            return;
        }

        const pageStart = state.inputPage * INPUT_PICKER_PAGE_SIZE;
        const pageItems = state.inputFiles.slice(pageStart, pageStart + INPUT_PICKER_PAGE_SIZE);
        if (state.activeKind === "images") {
            const grid = document.createElement("div");
            grid.className = "h3-media-input-grid";
            pageItems.forEach((path) => {
                const item = document.createElement("button");
                item.type = "button";
                item.className = "h3-media-input-item";
                item.classList.toggle("is-selected", state.inputSelection.has(path));
                item.title = path;
                item.addEventListener("click", () => toggleInputSelection(path));
                const image = document.createElement("img");
                image.loading = "lazy";
                image.decoding = "async";
                image.src = imagePreviewUrl(path);
                image.alt = displayName(path);
                const caption = document.createElement("span");
                caption.textContent = displayName(path);
                item.append(image, caption);
                grid.append(item);
            });
            picker.append(grid);
        } else {
            const list = document.createElement("div");
            list.className = "h3-media-input-file-list";
            pageItems.forEach((path) => {
                const item = document.createElement("label");
                item.className = "h3-media-input-file-item";
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = state.inputSelection.has(path);
                checkbox.addEventListener("change", () => toggleInputSelection(path));
                const type = document.createElement("span");
                type.textContent = iconForKind(state.activeKind);
                const name = document.createElement("span");
                name.textContent = displayName(path);
                name.title = path;
                item.append(checkbox, type, name);
                list.append(item);
            });
            picker.append(list);
        }
        content.append(picker);
    };

    const makeDraggable = (element, kind, index) => {
        element.draggable = true;
        element.addEventListener("dragstart", (event) => {
            state.drag = { kind, index };
            element.classList.add("is-dragging");
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/plain", `${kind}:${index}`);
        });
        element.addEventListener("dragend", () => {
            state.drag = null;
            element.classList.remove("is-dragging");
        });
        element.addEventListener("dragover", (event) => {
            if (state.drag?.kind !== kind) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
        });
        element.addEventListener("drop", (event) => {
            if (state.drag?.kind !== kind) return;
            event.preventDefault();
            reorder(kind, state.drag.index, index);
            state.drag = null;
        });
    };

    const renderImages = (items) => {
        const grid = document.createElement("div");
        grid.className = "h3-media-image-grid";
        items.forEach((path, index) => {
            const item = document.createElement("div");
            item.className = "h3-media-image-item";
            item.title = displayName(path);
            makeDraggable(item, "images", index);

            const image = document.createElement("img");
            image.loading = "lazy";
            image.draggable = false;
            image.src = imagePreviewUrl(path);
            image.alt = displayName(path);
            const remove = createIconButton("×", `删除第 ${index + 1} 张图片`, "h3-media-remove");
            remove.addEventListener("click", (event) => {
                event.stopPropagation();
                removeAt("images", index);
            });
            const number = document.createElement("div");
            number.className = "h3-media-order";
            number.textContent = String(index + 1);
            item.append(image, remove, number);
            grid.append(item);
        });
        content.append(grid);
    };

    const renderFileList = (kind, items) => {
        const list = document.createElement("div");
        list.className = "h3-media-file-list";
        items.forEach((path, index) => {
            const row = document.createElement("div");
            row.className = "h3-media-file-row";
            makeDraggable(row, kind, index);
            const number = document.createElement("span");
            number.className = "h3-media-file-order";
            number.textContent = String(index + 1);
            const type = document.createElement("span");
            type.className = "h3-media-file-type";
            type.textContent = iconForKind(kind);
            const name = document.createElement("span");
            name.className = "h3-media-file-name";
            name.textContent = displayName(path);
            name.title = path;
            const remove = createIconButton("×", `删除第 ${index + 1} 个${KIND_CONFIG[kind].singular}`, "h3-media-row-remove");
            remove.addEventListener("click", (event) => {
                event.stopPropagation();
                removeAt(kind, index);
            });
            row.append(number, type, name, remove);
            list.append(row);
        });
        content.append(list);
    };

    const render = () => {
        const manifest = currentManifest();
        tabs.replaceChildren();
        for (const [kind, config] of Object.entries(KIND_CONFIG)) {
            const tab = document.createElement("button");
            tab.type = "button";
            tab.className = "h3-media-tab";
            tab.classList.toggle("is-active", state.activeKind === kind);
            tab.textContent = `${config.label} ${manifest[kind].length}`;
            tab.addEventListener("click", () => {
                state.activeKind = kind;
                state.inputPickerOpen = false;
                state.inputFiles = [];
                state.inputSelection.clear();
                setStatus("");
                render();
            });
            tabs.append(tab);
        }

        const config = KIND_CONFIG[state.activeKind];
        addButton.title = `添加${config.singular}`;
        addButton.setAttribute("aria-label", addButton.title);
        addButton.disabled = state.uploading;
        inputButton.title = `从 input 目录选择${config.singular}`;
        inputButton.setAttribute("aria-label", inputButton.title);
        inputButton.disabled = state.uploading;
        clearButton.disabled = state.uploading || manifest[state.activeKind].length === 0;
        content.replaceChildren();
        if (state.inputPickerOpen) renderInputPicker();
        const items = manifest[state.activeKind];
        if (items.length === 0) {
            const empty = document.createElement("div");
            empty.className = "h3-media-empty";
            empty.textContent = `未选择${config.singular}`;
            content.append(empty);
        } else if (state.activeKind === "images") {
            renderImages(items);
        } else {
            renderFileList(state.activeKind, items);
        }
        node.setDirtyCanvas?.(true, true);
    };

    addButton.addEventListener("click", () => {
        if (state.uploading) return;
        const kind = state.activeKind;
        const config = KIND_CONFIG[kind];
        const input = document.createElement("input");
        input.type = "file";
        input.accept = config.accept;
        input.multiple = true;
        input.hidden = true;
        document.body.append(input);
        input.addEventListener("cancel", () => input.remove(), { once: true });
        input.addEventListener("change", async () => {
            const selected = Array.from(input.files || []);
            await addLocalFiles(selected, kind);
            input.remove();
        }, { once: true });
        input.click();
    });

    inputButton.addEventListener("click", async () => {
        if (state.uploading) return;
        state.inputPickerOpen = true;
        state.inputFiles = [];
        state.inputPage = 0;
        state.inputSelection.clear();
        setStatus(`正在读取 input/${KIND_CONFIG[state.activeKind].label}`);
        render();
        try {
            const response = await api.fetchApi(
                `/minimax_h3_easy/input-media?kind=${encodeURIComponent(state.activeKind)}`,
            );
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            state.inputFiles = Array.isArray(payload?.files) ? payload.files : [];
            state.inputPage = 0;
            setStatus(`找到 ${state.inputFiles.length} 项 input 媒体`);
        } catch (error) {
            state.inputFiles = [];
            setStatus(`读取 input 目录失败: ${error?.message || error}`, true);
        }
        render();
    });

    clearButton.addEventListener("click", () => {
        const manifest = currentManifest();
        manifest[state.activeKind] = [];
        commitManifest(node, manifest);
        setStatus("");
        render();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        content.addEventListener(eventName, (event) => {
            if (!event.dataTransfer?.files?.length || state.uploading) return;
            event.preventDefault();
            event.stopPropagation();
            content.classList.add("is-file-dragover");
        });
    });
    content.addEventListener("dragleave", (event) => {
        if (event.target === content) content.classList.remove("is-file-dragover");
    });
    content.addEventListener("drop", async (event) => {
        if (!event.dataTransfer?.files?.length || state.uploading) return;
        event.preventDefault();
        event.stopPropagation();
        content.classList.remove("is-file-dragover");
        await addLocalFiles(event.dataTransfer.files, state.activeKind);
    });

    const domWidget = node.addDOMWidget("h3_media_loader", "h3_media_loader", root, {
        serialize: false,
        hideOnZoom: false,
        canvasOnly: false,
        getMinHeight: () => 200,
        afterResize: () => node.setDirtyCanvas?.(true, true),
    });
    domWidget.serialize = false;
    domWidget.options ||= {};
    domWidget.options.serialize = false;
    domWidget.options.hideOnZoom = false;
    domWidget.options.canvasOnly = false;
    node.__h3MediaLoaderWidget = domWidget;
    node.__h3MediaLoaderRoot = root;
    node.__h3MediaLoaderResizeStatus = resizeStatus;
    node.__h3MediaLoaderRender = render;
    refreshResizeControls(node);
    render();
}

function installStyles() {
    if (document.getElementById("h3-media-loader-styles")) return;
    const style = document.createElement("style");
    style.id = "h3-media-loader-styles";
    style.textContent = `
        .h3-media-loader { width:100%; height:100%; min-width:0; min-height:0; box-sizing:border-box; display:flex; flex-direction:column; gap:6px; overflow:hidden; color:var(--input-text, #ddd); font:12px sans-serif; }
        .h3-media-resize-status { flex:0 0 auto; min-width:0; padding:6px 8px; overflow:hidden; border:1px solid var(--border-color, #444); border-radius:5px; background:var(--comfy-input-bg, #222); color:var(--input-text, #ddd); font-variant-numeric:tabular-nums; text-overflow:ellipsis; white-space:nowrap; }
        .h3-media-tabs { flex:0 0 auto; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:2px; padding:2px; background:var(--comfy-input-bg, #222); border:1px solid var(--border-color, #444); border-radius:6px; }
        .h3-media-tab { min-width:0; height:26px; padding:0 5px; border:0; border-radius:4px; background:transparent; color:inherit; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .h3-media-tab.is-active { background:var(--comfy-menu-bg, #333); box-shadow:inset 0 0 0 1px var(--border-color, #555); }
        .h3-media-toolbar { flex:0 0 auto; display:grid; grid-template-columns:28px 28px 28px minmax(0,1fr); gap:5px; align-items:center; min-height:28px; }
        .h3-media-icon-button { width:28px; height:28px; padding:0; border:1px solid var(--border-color, #555); border-radius:5px; background:var(--comfy-input-bg, #222); color:inherit; cursor:pointer; font-size:18px; line-height:1; }
        .h3-media-icon-button:disabled { cursor:default; opacity:.4; }
        .h3-media-icon-button:hover:not(:disabled), .h3-media-tab:hover { filter:brightness(1.18); }
        .h3-media-status { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; opacity:.78; }
        .h3-media-status.is-error { color:#ff8d83; opacity:1; }
        .h3-media-content { flex:1 1 auto; min-width:0; min-height:0; overflow:auto; padding:6px; border:1px solid var(--border-color, #444); border-radius:6px; background:var(--comfy-input-bg, #202020); }
        .h3-media-content.is-file-dragover { border-color:var(--input-text, #ddd); background:var(--comfy-menu-bg, #2b2b2b); }
        .h3-media-empty { width:100%; height:100%; min-height:72px; display:flex; align-items:center; justify-content:center; opacity:.58; }
        .h3-media-image-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(76px,1fr)); gap:7px; align-content:start; }
        .h3-media-image-item { position:relative; display:grid; grid-template-rows:minmax(68px,1fr) 20px; min-width:0; aspect-ratio:4/5; overflow:hidden; border:1px solid var(--border-color, #555); border-radius:6px; background:var(--comfy-menu-bg, #2b2b2b); cursor:grab; }
        .h3-media-image-item.is-dragging, .h3-media-file-row.is-dragging { opacity:.48; }
        .h3-media-image-item img { width:100%; height:100%; min-height:0; object-fit:cover; display:block; background:#111; }
        .h3-media-order { display:flex; align-items:center; justify-content:center; min-width:0; font-variant-numeric:tabular-nums; font-weight:600; }
        .h3-media-remove { position:absolute; top:4px; right:4px; width:22px; height:22px; border-color:rgba(255,255,255,.35); background:rgba(20,20,20,.78); }
        .h3-media-file-list { display:flex; flex-direction:column; gap:5px; }
        .h3-media-file-row { display:grid; grid-template-columns:28px 48px minmax(0,1fr) 28px; gap:5px; align-items:center; min-height:32px; padding:3px; border:1px solid var(--border-color, #4b4b4b); border-radius:5px; background:var(--comfy-menu-bg, #2b2b2b); cursor:grab; }
        .h3-media-file-order { text-align:center; font-variant-numeric:tabular-nums; font-weight:600; }
        .h3-media-file-type { font-size:9px; opacity:.65; }
        .h3-media-file-name { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .h3-media-row-remove { width:24px; height:24px; justify-self:end; }
        .h3-media-input-picker { display:flex; flex-direction:column; gap:6px; margin-bottom:8px; padding-bottom:7px; border-bottom:1px solid var(--border-color, #444); }
        .h3-media-input-picker-header { display:flex; align-items:center; justify-content:space-between; gap:6px; min-width:0; font-weight:600; }
        .h3-media-input-picker-header > span { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .h3-media-input-picker-actions { display:flex; flex:0 0 auto; gap:4px; }
        .h3-media-input-page { min-width:34px; align-self:center; text-align:center; font-variant-numeric:tabular-nums; opacity:.75; }
        .h3-media-input-empty { min-height:40px; display:flex; align-items:center; justify-content:center; opacity:.6; }
        .h3-media-input-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(72px,1fr)); gap:6px; }
        .h3-media-input-item { position:relative; display:grid; grid-template-rows:58px 18px; min-width:0; padding:2px; border:1px solid var(--border-color, #555); border-radius:5px; background:var(--comfy-menu-bg, #2b2b2b); color:inherit; cursor:pointer; overflow:hidden; }
        .h3-media-input-item.is-selected { border-color:var(--input-text, #ddd); box-shadow:inset 0 0 0 2px var(--input-text, #ddd); }
        .h3-media-input-item img { width:100%; height:100%; min-height:0; object-fit:cover; display:block; background:#111; }
        .h3-media-input-item span { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:10px; line-height:18px; }
        .h3-media-input-file-list { display:flex; flex-direction:column; gap:4px; }
        .h3-media-input-file-item { display:grid; grid-template-columns:18px 42px minmax(0,1fr); gap:5px; align-items:center; min-width:0; padding:4px; border:1px solid var(--border-color, #555); border-radius:4px; cursor:pointer; }
        .h3-media-input-file-item span:last-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    `;
    document.head.append(style);
}

app.registerExtension({
    name: "MiniMaxH3Easy.MediaLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        installStyles();

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedH3MediaLoader() {
            const result = originalCreated?.apply(this, arguments);
            setupMediaLoader(this);
            this.setSize?.([420, 460]);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigureH3MediaLoader(info) {
            const result = originalConfigure?.apply(this, arguments);
            setupMediaLoader(this);
            this.__h3MediaLoaderRender?.();
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onRemovedH3MediaLoader() {
            this.__h3MediaLoaderRoot?.remove?.();
            this.__h3MediaLoaderWidget = null;
            this.__h3MediaLoaderRoot = null;
            this.__h3MediaLoaderResizeStatus = null;
            this.__h3MediaLoaderRender = null;
            return originalRemoved?.apply(this, arguments);
        };
    },
});
