# Hugging Face model I/O report

Inspection date: 2026-05-31

Authenticated as Hugging Face user `mrive052` with a read token.

## Access status

| Model | Access to metadata | Access to config/checkpoints | Notes |
|---|---:|---:|---|
| `facebook/sam2.1-hiera-large` | yes | yes | Transformers integration available. |
| `facebook/sam3` | yes | no, 403 gated | README visible, config/tokenizer/processor gated. Accept terms/request access. |
| `facebook/sam3.1` | yes | no, 403 gated | Checkpoint-only repo; README says no Transformers integration. Accept terms/request access. |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | yes | no, 403 gated | README visible, config/preprocessor gated. Accept terms/request access. |
| `nvidia/LocateAnything-3B` | yes | yes | Transformers custom code, needs `trust_remote_code=True`. |
| `Ultralytics/YOLO26` | yes | yes | Ultralytics `.pt` weights, not standard Transformers schema. |
| `Roboflow/rf-detr-segmentation` | yes | yes | Transformers integration available. |
| `facebook/sapiens2-pose-5b` | yes | yes | Sapiens library checkpoint; use linked Sapiens2 repo. |
| `facebook/sam-3d-objects` | yes | yes | SAM 3D Objects library/checkpoints; use linked GitHub code. |
| `facebook/sam-3d-body-dinov3` | yes | no, 403 gated for config | README visible, model config/checkpoints gated. Accept terms/request access. |

## Per-model I/O

### `facebook/sam2.1-hiera-large`

- HF task: `mask-generation`
- Library: `transformers`
- Model type: `sam2_video`
- Architecture: `Sam2VideoModel`
- Processor: `Sam2VideoProcessor`
- Image processor: `Sam2ImageProcessorFast`
- Preprocessing:
  - resize to `1024 x 1024`
  - rescale by `1 / 255`
  - normalize with ImageNet mean/std
  - mask size `256 x 256`

Inputs:

- Automatic mask generation pipeline:
  - image URL/path/PIL image
  - optional `points_per_batch`
- Image prompt segmentation:
  - `images`
  - `input_points`: shape convention `[image, object, point, xy]`
  - `input_labels`: positive `1`, negative `0`
  - optional `input_boxes`: XYXY boxes, shape `[image, box, 4]`
  - optional `input_masks` for refinement
  - optional previous `image_embeddings`
- Video tracking:
  - video frames / initialized video session
  - prompts: points, labels, boxes
  - optional streaming frame tensor

Outputs:

- Image:
  - `pred_masks`
  - `iou_scores`
  - `image_embeddings`
  - postprocessed masks via `processor.post_process_masks(...)`
- Pipeline:
  - dictionary containing `masks`
- Video:
  - per-frame `pred_masks`
  - frame index
  - object IDs

### `facebook/sam3`

- HF task: `mask-generation`
- Library: `transformers`
- Access: gated for config/processor/tokenizer files with current token.

From visible README:

Inputs:

- Image PCS, promptable concept segmentation:
  - `images`
  - `text` prompt, e.g. object/concept name
  - optional `input_boxes`
  - optional `input_boxes_labels`, where `1` is positive and `0` is negative
  - batched mixed prompts supported: some images can use text, others boxes
- Tracker/PVS mode:
  - points, boxes, masks similar to SAM2
- Video PCS/PVS:
  - video session or streaming frame
  - text prompt and/or visual prompts

Outputs:

- Image PCS:
  - instance masks
  - boxes
  - scores
  - semantic segmentation tensor `semantic_seg`
  - postprocessed results via `processor.post_process_instance_segmentation(...)`
- Video:
  - per-frame object IDs
  - scores
  - boxes in absolute XYXY pixel coordinates
  - masks at original resolution

Action needed for exact config/signature: accept/request access on the model page, then rerun inspection.

### `facebook/sam3.1`

- HF task: `mask-generation`
- Library name: `checkpoint`
- Access: gated for config/processor/tokenizer files with current token.
- README says this repo hosts only SAM 3.1 checkpoints and has no Hugging Face Transformers integration.

Expected I/O from README summary:

- Inputs:
  - images or videos
  - text prompts
  - visual prompts: points, boxes, masks
- Outputs:
  - masks
  - boxes
  - tracking outputs for videos

Action needed: accept/request access and use the SAM3 GitHub repository for exact callable API.

### `facebook/dinov3-vitl16-pretrain-lvd1689m`

- HF task: `image-feature-extraction`
- Library: `transformers`
- Access: gated for config/preprocessor files with current token.

From visible README:

Inputs:

- image / batch of images
- processor returns `pixel_values`
- ViT-L/16 accepts image shapes that are multiples of patch size `16`; otherwise crops to nearest smaller multiple.

Outputs:

- class token
- patch tokens
- register tokens
- `pooler_output` when using `AutoModel`

For a `224 x 224` image:

- 1 class token
- 4 register tokens
- 196 patch tokens

Action needed for exact config: accept/request access.

### `nvidia/LocateAnything-3B`

- HF task: `image-text-to-text`
- Library: `transformers`
- Custom code: yes, requires `trust_remote_code=True`
- Model type: `locateanything`
- Architecture: `LocateAnythingForConditionalGeneration`
- Processor: `LocateAnythingProcessor`
- Image processor: `LocateAnythingImageProcessor`
- Tokenizer: `Qwen2Tokenizer`
- Image normalization: mean/std `[0.5, 0.5, 0.5]`

Inputs:

- RGB image at original source resolution
- text prompt / chat-style message
- processor call produces:
  - `input_ids`
  - `attention_mask`
  - `pixel_values`
  - optional `image_grid_hws`
- Supports image and video placeholders in processor code.
- Production image resolution supports up to about `2.5K` per README.
- Generation supports up to `8192` newly generated tokens per README.

Outputs:

- Text output from generation.
- Output is structured into fixed-length blocks of length 6:
  - semantic blocks
  - box blocks
  - negative blocks
  - end blocks
- Coordinates are normalized integers in `[0, 1000]` and must be scaled to pixel coordinates.
- README worker exposes parsers for:
  - boxes: `x1, y1, x2, y2`
  - points: `x, y`

Supported task families from README:

- object detection
- phrase grounding
- multi-object grounding
- scene text detection
- GUI grounding
- pointing
- layout/OCR localization

### `Ultralytics/YOLO26`

- Library: `ultralytics`
- Files: multiple `.pt` weights for tasks and sizes:
  - detection: `yolo26{n,s,m,l,x}.pt`
  - segmentation: `yolo26{n,s,m,l,x}-seg.pt`
  - classification: `yolo26{n,s,m,l,x}-cls.pt`
  - pose: `yolo26{n,s,m,l,x}-pose.pt`
  - oriented boxes: `yolo26{n,s,m,l,x}-obb.pt`

Inputs:

- image path, URL, PIL image, OpenCV/numpy image, video, stream, etc. through Ultralytics `YOLO(...)` API
- common inference args include `imgsz`, confidence threshold, device, etc.

Outputs depend on selected checkpoint:

- Detection:
  - boxes
  - class IDs/names
  - confidences
- Segmentation:
  - boxes
  - classes/confidences
  - instance masks
- Classification:
  - class probabilities
- Pose:
  - boxes
  - person class/confidence
  - keypoints
- OBB:
  - oriented bounding boxes

Exact schema source: Ultralytics `Results` object docs, not HF config JSON.

### `Roboflow/rf-detr-segmentation`

- HF task: `image-segmentation`
- Library: `transformers`
- Model type: `rf_detr`
- Architecture: `RfDetrForInstanceSegmentation`
- Image processor: `RfDetrImageProcessor`
- Number of queries: `200`
- Classes: COCO-style labels in `id2label`
- Preprocessing:
  - resize to `432 x 432`
  - rescale by `1 / 255`
  - normalize with ImageNet mean/std

Inputs:

- `images`
- processor returns tensors for model inference, primarily `pixel_values`

Outputs:

- raw model outputs for instance segmentation
- postprocess with `processor.post_process_instance_segmentation(outputs, target_sizes=..., threshold=...)`
- final outputs:
  - per-instance masks
  - bounding boxes
  - class labels
  - class scores

### `facebook/sapiens2-pose-5b`

- HF task: `keypoint-detection`
- Library: `sapiens`
- Checkpoint: `sapiens2_5b_pose.safetensors`
- Inference resolution from README: `1024 x 768` `(H x W)`

Inputs:

- human-centric imagery
- top-down pose pipeline from the Sapiens2 repository
- likely person crops or detected person regions, depending on demo config

Outputs:

- 308 keypoints following the Sociopticon keypoint format:
  - detailed face: 274 keypoints
  - plus hand/foot/body keypoints
- Visualization/output options are in Sapiens2 `docs/POSE.md`.

Exact schema source: Sapiens2 repository configs and pose demo scripts, not HF model config.

### `facebook/sam-3d-objects`

- Library: `sam-3d-objects`
- Task: single/multi-object 3D reconstruction from images
- Repo includes pipeline/checkpoint YAMLs and checkpoint files.

Inputs from README example:

- RGBA image, with mask embedded in alpha channel, or image plus separate mask
- object mask, loaded with `load_single_mask(...)`
- optional seed

Example API:

```python
output = inference(image, mask, seed=42)
```

Outputs:

- 3D object representation including Gaussian splat output:
  - `output["gs"]`
  - can export with `output["gs"].save_ply("splat.ply")`
- Other outputs depend on the SAM 3D Objects inference code/pipeline config, likely mesh/layout/texture-related artifacts.

Exact schema source: `facebookresearch/sam-3d-objects` GitHub `Inference` class.

### `facebook/sam-3d-body-dinov3`

- Library: `sam-3d-body`
- Access: README visible, `model_config.yaml` gated with current token.
- Task: single-image full-body 3D human mesh recovery.

Inputs from README:

- RGB image
- optional auxiliary prompts:
  - 2D keypoints
  - masks

Example API:

```python
outputs = estimator.process_one_image(rgb_image)
```

Outputs listed in README:

- `pred_vertices`: 3D mesh vertices in camera coordinates
- `pred_keypoints_3d`: 3D pose keypoints
- `pred_keypoints_2d`: 2D pose keypoints projected to image

Action needed for exact config/checkpoint schema: accept/request access on the model page.
