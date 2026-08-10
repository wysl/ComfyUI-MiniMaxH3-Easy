# ComfyUI-MiniMaxH3-Easy

[中文说明 / Chinese documentation](README_CN.md)

`ComfyUI-MiniMaxH3-Easy` integrates MiniMax H3 text-to-video, image-to-video,
and reference-to-video generation into one streamlined ComfyUI workflow surface.
The interaction layer has been deliberately polished to make media input,
reference selection, and prompt editing simple to understand and quick to use.

The idea is simple: keep the power of ComfyUI, while removing the repetitive
media wiring and reference bookkeeping that normally make MiniMax H3 workflows
hard to read and harder to learn.

## Updates

### 2026-08-10

- `H3 Context` now exposes the exact video duration to Prompt Assistant for
  multi-shot timing in T2VA, I2VA, FL2VA, and L2VA prompts.
- Fixed connected `Seconds` values being overwritten by the node's stale internal
  widget value, so one duration input can now drive both H3 generation and video saving.

### 2026-08-09

- Added **MiniMax H3 Easy Save Video**, whose preview follows manual node resizing
  without expanding to the video's native dimensions after execution.
- Added an `IMAGE` batch containing the first frame of every second, calculated
  from the connected duration and FPS without requiring a manual frame count.
- Added a separate `Last frame` `IMAGE` output taken directly from the end of the
  actual decoded video frame sequence.

## Highlights

### One `Media` input for mixed media

The main node uses one visible `Media` input for images, videos, and audio.
Multiple links can enter the same port. Image, video, and audio order numbers
are tracked independently, and each media type has its own wire color and
preview style.

<p align="center">
  <img src="images/mixed-media-input-en.png" alt="Mixed media input" width="560">
</p>

This keeps the graph compact without losing ordering information. Drag from
`Media` to an empty area of the canvas to quickly create a compatible media
node. Click the number in the middle of a virtual media wire to open the small
delete menu.

<p align="center">
  <img src="images/quick-create-node-en.png" alt="Quick-create media node" width="460">
</p>

### A complete `@` reference editor

`@` is available in **Reference Video** mode. Type `@` to select a connected
image, video, or standalone audio resource. The popup presents images first,
videos second, and audio last, with a preview for each item.

<p align="center">
  <img src="images/mention-popup-en.png" alt="Reference popup" width="320">
</p>

<p align="center">
  <img src="images/reference-editor-en.png" alt="Reference editor" width="720">
</p>

References use **By index** by default because it is concise and easy to scan.
**By filename** is available when the filename itself is more meaningful.

The chips are an editing interface only. When the workflow runs, the node
automatically converts them into the reference format recommended by MiniMax,
including `<Picture N>`, `<Video N>`, and `<Audio N>`. Users do not need to
manually write or maintain those tags.

A video's synchronized soundtrack is handled together with that video. A
standalone audio input remains an independent reference, so users only need to
connect the media they actually want to use.

### Simple dialogue blocks

Type `#` in the prompt editor to insert an editable dialogue block.

<p align="center">
  <img src="images/dialogue-block-en.png" alt="Dialogue block" width="560">
</p>

- Press `Enter` to finish the block.
- Press `Shift+Enter` to add a line break inside it.
- Click the block at any time to edit it again.

The block is automatically converted to MiniMax's recommended dialogue format
`<d>...</d>` when the prompt is sent. The rest of the prompt stays ordinary
prompt text, so users can describe the scene naturally without learning the
underlying markup.

## Nodes and connections

### MiniMax H3 Easy Loader

The all-in-one loader exposes separate choices for:

- FL2VA model;
- Ref2VA model;
- Qwen3-VL text encoder;
- video VAE;
- audio VAE.

Official and common community filename variants are recognized, including
BF16, FP8, INT8, INT4, NVFP4, NF4, and GGUF releases.

To use only one transformer model, set the other model selector to `None`. The
remaining model will automatically be used for text-to-video,
I2V/first-last-frame, and reference-video generation. When both models are
available, the node prefers FL2VA for text-to-video and I2V/first-last-frame
generation, and Ref2VA for reference-video generation. At least one of the two
transformer models must be selected.

### MiniMax H3 Easy

This is the main generation node. It outputs:

- `Model` — connect this to a model-only LoRA, Sage Attention patch, or directly
  to the sampler;
- `H3 Context` — connect this to **MiniMax H3 Easy Output**.

### MiniMax H3 Easy Output

This node expands `H3 Context` into the standard workflow outputs:

- Conditioning;
- Latent;
- Video VAE;
- Audio VAE;
- FPS.

The output node also has an optional `Optimized prompt` input. For Prompt
Assistant integration, connect the workflow as follows:

1. Fan out `MiniMax H3 Easy.H3 Context` to both
   `Multimedia Reference Fusion Prompt.H3 Context` and
   `MiniMax H3 Easy Output.H3 Context`.
2. Connect `Multimedia Reference Fusion Prompt.Fusion Prompt` to
   `MiniMax H3 Easy Output.Optimized prompt`.
3. The fusion node reuses the original prompt, mode, and ordered media; the
   output node then re-encodes Conditioning with the optimized result.

When `Optimized prompt` is not connected, the original Conditioning is used
unchanged, preserving existing workflows.

### MiniMax H3 Easy Save Video

Connect a ComfyUI `VIDEO` to this node to save it. The custom preview has a small
fixed minimum size and follows manual node width and height changes. Loading the
saved video never resizes the node to the video's native resolution.

The node accepts the video duration and FPS, along with the filename prefix,
container format, and codec. Its `FPS` input can connect directly to the matching
output on **MiniMax H3 Easy Output**. It exposes three outputs:

- `Video` passes the saved input `VIDEO` through for downstream use.
- `First frame of each second` is an `IMAGE` batch sampled at each whole-second
  boundary.
- `Last frame` is a single `IMAGE` taken directly from the end of the decoded
  frame sequence.

Per-second indexes use `floor(second index × FPS)`. For a 5-second, 24 FPS video
with 121 frames, the batch contains frames `0, 24, 48, 72, 96`, while `Last
frame` returns frame `120`. The node reads the actual frame count automatically
and uses it to prevent out-of-range extraction.

The sampler, acceleration nodes, and other video/audio processing nodes remain
outside the main node so the workflow stays compatible with the rest of ComfyUI.

## Modes

### I2V or First/Last Frame

- No media connected: text-to-video.
- One image connected: image-to-video.
- Two images connected: first/last-frame generation.
- Video and audio links are not accepted in this mode.

### Reference Video

- Up to nine reference images, three reference videos, and three standalone
  audio clips.
- The `@` editor is enabled.
- Reference order and prompt references are kept synchronized automatically.

## Parameter design

### Resolution and aspect ratio

Resolution presets follow the MiniMax H3/ComfyUI megapixel-style budgets:

`360P`, `416P`, `480P`, `540P`, `640P`, `720P`, `768P`, `832P`, `928P`,
`1024P`, `1080P`, and `Custom`.

Presets calculate the canvas from the selected aspect ratio and align the final
dimensions to multiples of 32. Available ratios are `1:1`, `2:3`, `3:2`, `3:4`,
`4:3`, `9:16`, `16:9`, and `21:9`.

Selecting `Custom` reveals width and height and hides the aspect-ratio control.
Custom width and height must be multiples of 32.

### Duration

Duration is set in seconds from **4 to 20**. The requested duration is aligned
to MiniMax H3's frame rules internally.

### Advanced options

Advanced options are off by default. When enabled, they reveal:

- first/last-frame setup in I2V or First/Last Frame mode;
- reference image size in Reference Video mode;
- `@` display mode in Reference Video mode.

Reference images use a short-edge limit of **1K** or **2K**. Images below the
limit keep their original resolution; larger images are resized proportionally
instead of being forced down to the output video's resolution.

## Installation and models

Install this directory as:

```text
ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-Easy
```

Place models in the standard folders:

```text
ComfyUI/models/diffusion_models/
ComfyUI/models/text_encoders/
ComfyUI/models/vae/
```

For `.gguf` transformer or text-encoder files, install
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and restart ComfyUI.
GGUF files are routed automatically to their GGUF loader; regular safetensors
files continue to use native ComfyUI loading.

## License and attribution

This project is released under the [MIT License](LICENSE). 

If you reference, reuse, or adapt a substantial part of this project, please
credit the original author and mention `ComfyUI-MiniMaxH3-Easy` in your project
documentation.

Please do not present the project's multi-media input design, `@` reference
editor, dialogue-block conversion, or related implementation as entirely your
own work.

## Important notes

- I2V or First/Last Frame mode accepts at most two images.
- Reference Video mode accepts at most nine images, three videos, and three
  standalone audio clips.
- A video's synchronized audio is paired with that video automatically and does
  not consume a separate audio slot.
- Image, video, and audio numbering is independent.
- The save node's `FPS` input can connect directly to the matching output on
  **MiniMax H3 Easy Output**.
- The node supports both the legacy ComfyUI canvas and Nodes 2.0.
- Chinese browsers show Chinese parameter labels; other browsers show English
  labels.
- Model-only LoRA and attention/acceleration patches connect after the main
  node's `Model` output.
