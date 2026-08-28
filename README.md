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

### 2026-08-23

- Improved **MiniMax H3 Easy Media Loader** scaling controls: the resize mode is now the first visible option, and only the active scale, long-edge, or short-edge controls remain visible.
- Added an `input` directory browser with category-specific media selection and image thumbnails with ordering.
- Local image, audio, and video files can now be dropped directly onto the active media area and are uploaded into the manifest.
- Added a dedicated `Concatenated image` output that vertically combines selected images in order while preserving the existing `multi output` image list.

### 2026-08-26

- The **MiniMax H3 Easy Media Loader** input browser now paginates large image folders,
  rendering at most 48 cards at a time while keeping selections across pages.
- Image cards use a lightweight 256px server thumbnail and asynchronous browser decoding,
  avoiding full-resolution image downloads and browser stalls when the input folder is large.
- **MiniMax H3 Easy Video Black Intro** now falls back automatically on older ComfyUI VIDEO
  APIs that do not accept the optional `color_space` constructor argument.

### 2026-08-28

- Added **MiniMax H3 Easy Swap Dimensions**, which passes connected `width` and `height`
  values through and optionally exchanges them with the `交换长宽` switch. Its outputs
  keep the standard `width`/`height` names for direct connection to resize nodes.
- Added **MiniMax H3 Easy Video Black Intro**, which replaces the first user-specified
  number of seconds with black frames while preserving the source FPS, audio, resolution,
  metadata, alpha channel, bit depth, and color space.
- The media loader now defaults to `0.5` for scale mode and `1024` pixels for long-edge/short-edge modes; image resizing always uses `lanczos`.
- Added automatic repair for the legacy resize-widget order so older workflows do not pass `lanczos` or `long edge` into the wrong parameter.

### 2026-08-21

- Fixed **MiniMax H3 Easy Second Pass Conditioning** for I2V first-frame and first/last-frame workflows: after a latent upscale it now rebuilds both boundary keyframes at the padded H3 patch grid, including odd latent widths or heights.
- Older serialized H3 Contexts without source pixels now resize their stored keyframe latents instead of silently reusing the old resolution, preventing `condition video` token-count mismatches in `SamplerCustomAdvanced`.

### 2026-08-17

- Distant Face Refine now defaults to identity-only prompting when an identity image is connected, preventing later scenes' camera, environment, and lighting instructions from changing facial geometry, skin tone, or exposure.
- Identity references now derive a stable refinement seed across chained segments. Full-scene prompting and the supplied input seed remain available as compatibility modes.
- Added **MiniMax H3 Easy Chroma Context**, which extracts the final `1 / 5 / 22 / 39` clean frames and creates a separate tapered chroma-noise context video for the next H3 Motion Context segment.
- Clean and noisy context outputs remain separate, so chroma noise is never written into the delivered or assembled video.
- Added optional per-pixel Rec.709 luminance preservation, enabled by default, so chroma noise keeps each context frame's original lighting without flattening motion or exposure changes.
- Added **MiniMax H3 Easy Seam Stabilizer** for long-video chains. It estimates the expected camera motion from the overlap tail, limits an abnormal boundary shift, and fades low-frequency local colour correction across the first generated frames.
- Seam stabilization keeps the overlap frame count unchanged, uses a cosine fade that reaches exactly zero, and adds no Python dependency.
- The node uses the existing PyTorch and ComfyUI VIDEO APIs and adds no Python dependency.

### 2026-08-16

- Added **MiniMax H3 Easy Distant Face Refine**, which tracks small faces per frame,
  regenerates stabilised face crops with H3, and stitches them back with temporal
  smoothing, colour matching, and feathered masks.
- The node supports one or two people, preserves the source video's audio, FPS,
  metadata, alpha channel, and bit depth, and expands into ordinary ComfyUI nodes
  so model caching and VRAM management remain visible to the execution engine.
- H3 Turbo LoRA step counts are detected from names such as `4step` and `8step`;
  manual step selection remains available.

### 2026-08-13

- Added **MiniMax H3 Easy Frame Interpolation**, which uses WhiteRabbit RIFE 4.7 to double the input video's FPS.
- The node reads the source FPS automatically and preserves the original resolution, audio, and video bit depth without upscaling.
- It uses `scale_factor=1.0` with `ensemble` disabled for a fast, resource-conscious quality balance. The `comfyui-WhiteRabbit` custom node must be installed and enabled.

### 2026-08-10

- `H3 Context` now exposes the exact video duration to Prompt Assistant for
  multi-shot timing in T2VA, I2VA, FL2VA, and L2VA prompts.
- Fixed connected `Seconds` values being overwritten by the node's stale internal
  widget value, so one duration input can now drive both H3 generation and video saving.
- Fixed some durations being aligned downward. Frame counts now round up to H3's
  `17k+5` grid so generated videos are never shorter than the requested duration.

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

### MiniMax H3 Easy Media Loader

The media loader keeps images, audio, and video from the `input` directory in separate lists.
Choose one resize mode first; only the parameters used by that mode remain visible. Existing
media can be multi-selected from the `input` browser, and local files can be dropped directly
onto the active category. Images show thumbnails and their selection order.

The `multi output`, `audio output`, and `video output` ports remain independent. When the dedicated
`Concatenated image` port is connected, selected images are combined vertically in order into one
`IMAGE`; narrower images receive black right-side padding and the original image list is unchanged.

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
Both transformer selectors expose every weight whose relative filename contains
`h3`, so community exports without an explicit FL2VA or Ref2VA suffix remain selectable.

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
- FPS;
- the original width and height actually used by H3;
- scaled width and height using `1.2 / 1.4 / 1.5 / 1.6 / 2.0`, aligned to 32.

The scaled dimensions can connect directly to **Resize Image v2**. To apply
both target dimensions exactly, set that node's `keep_proportion` to `stretch`
and `divisible_by` to `32`.

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

### MiniMax H3 Easy Distant Face Refine

Connect the generated `VIDEO`, the same loader `H3 Bundle`, and the matching
generation `H3 Context`. The node detects and stabilises distant faces, performs
a local H3 img2img pass on the crops, and returns a `VIDEO` with its original
audio, FPS, metadata, alpha channel, and bit depth intact.

- Run it on the original `24 FPS` H3 video, before frame interpolation.
- `single person` works without an identity image when the intended subject is
  the only or largest face. An optional identity reference locks tracking in crowds.
- `two people` requires both identity reference inputs. Each person is refined in
  sequence so the first composite is retained while the second person is processed.
- `steps=0` detects `4step` or `8step` from the selected Turbo LoRA filename. It
  falls back to 20 steps without a Turbo LoRA.
- With an identity reference connected, `prompt_mode` defaults to
  `identity only (recommended)` and excludes scene, camera, and lighting changes
  from the local face redraw.
- `seed_mode` defaults to `identity locked (recommended)`, deriving the same
  refinement seed from the identity image across chained segments.
- Select `full scene prompt` and `input seed` to restore the previous behaviour.
  Prompt Assistant's `Optimized prompt` scene description is only submitted in
  full-scene mode.

Required runtime assets:

- `face_yolov8m.pt` under `models/ultralytics/bbox/` (or another detector shown
  by the node);
- a Ref2VA transformer selected in **MiniMax H3 Easy Loader**;
- an H3 Turbo LoRA is recommended for speed but not mandatory;
- InsightFace `buffalo_l` is used for identity tracking and may be downloaded on
  first use. Install only one ONNX Runtime variant to avoid provider conflicts.

This feature is adapted from
[`ComfyUI-H3-FaceRefine`](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)
under the MIT License. See `THIRD_PARTY_NOTICES.md`.

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

### MiniMax H3 Easy Video Black Intro

Connect a ComfyUI `VIDEO` to this node and enter the number of seconds to replace at
the beginning. The node reads the input video's real FPS and replaces
`ceil(seconds × FPS)` frames with full black frames. It keeps the total frame count,
duration, resolution, audio track, metadata, alpha channel, bit depth, and color space
unchanged. A value of `0` passes the original video through unchanged. If the requested
duration is longer than the video, the complete video becomes black.

Audio is intentionally preserved and is not muted during the black interval. If a silent
black opening is needed, mute or trim the audio separately in the surrounding workflow.

### MiniMax H3 Easy Swap Dimensions

Connect any integer width and height outputs to this node. With `保持原顺序`, the two
outputs remain `width` and `height`; enabling `交换长宽` sends the original height to the
`width` output and the original width to the `height` output. This is useful when a
workflow needs to rotate the target orientation before `Resize Image v2` without adding
manual integer fields.

### MiniMax H3 Easy Second Pass Conditioning

Place this node between **H3 Context** and the second-pass guider. Connect the
video-only 24-channel latent after the spatial latent upscaler to its second
input. It synchronizes first-frame and first/last-frame conditioning with the
upscaled latent before `SamplerCustomAdvanced`. The node also handles older
contexts that retained encoded keyframes but no longer have the original image
pixels.

### MiniMax H3 Easy Frame Interpolation

Connect any ComfyUI `VIDEO` to this node to double its frame rate with WhiteRabbit
RIFE 4.7. The source FPS is detected automatically, so a `24 FPS` input produces a
`48 FPS` output without manual frame-count or frame-rate settings.

- The node does not resize images or latents.
- Original frames remain unchanged, with one generated frame inserted between each pair.
- An `N`-frame input produces `2N` frames: it inserts one frame between each pair and holds the final source frame once, keeping the encoded video and audio duration unchanged.
- Audio, metadata, and video bit depth are retained.
- The new FPS is also exposed as a `FLOAT` output for save nodes.
- RIFE 4.7, `scale_factor=1.0`, and `ensemble=false` are fixed internally.

No new Python dependency is added, but
[`comfyui-WhiteRabbit`](https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit)
must be installed and enabled with its `rife47.pth` model available. The node
provides an actionable error when this dependency is missing.

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

Duration is set in seconds from **4 to 20**. MiniMax H3 requires frame counts on
the `17k+5` grid, so the node rounds up to the nearest valid count. The actual
duration may be slightly longer than requested, but never shorter.

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
