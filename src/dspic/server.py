"""MVP server CLI for hosting normalized DSPIC vision models.

This module intentionally does not load model weights at import time. Models are
loaded only by `dspic-server serve ...` on the target machine.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from dspic.foundation_vision_types import (
    BBox,
    BoxOutput,
    Detection,
    ImageInput,
    MaskInstance,
    MaskOutput,
    MaskRef,
    ModelUsage,
    Point2D,
    PointDetection,
    PointOutput,
    RawTextOutput,
    VisionRequest,
    VisionResponse,
)


class VisionWorker(ABC):
    @abstractmethod
    def predict(self, request: VisionRequest) -> VisionResponse: ...


def _image_from_input(input_: ImageInput):
    """Load a PIL image from a normalized ImageInput."""

    import httpx
    from PIL import Image

    ref = input_.image
    if ref.path is not None:
        return Image.open(ref.path).convert("RGB")
    if ref.url is not None:
        response = httpx.get(ref.url, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    if ref.data is not None:
        data = ref.data
        if isinstance(data, str):
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            data = base64.b64decode(data)
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError("ImageRef must use path, url, or base64 data on the server.")


def _first_image(request: VisionRequest) -> ImageInput:
    for input_ in request.inputs:
        if isinstance(input_, ImageInput):
            return input_
    raise ValueError("Request requires an image input.")


def _prompt_text(request: VisionRequest) -> str:
    texts = [input_.text for input_ in request.inputs if input_.type == "text_prompt"]
    if texts:
        return "\n".join(texts)
    raise ValueError("Request requires a text_prompt input.")


class LocateAnythingWorker(VisionWorker):
    def __init__(self, model: str, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

    def predict(self, request: VisionRequest) -> VisionResponse:
        import torch

        image = _image_from_input(_first_image(request))
        prompt = _prompt_text(request)
        generation_mode = request.config.extensions.get("generation_mode", "hybrid")
        max_new_tokens = request.config.extensions.get("max_new_tokens", 2048)
        temperature = request.config.extensions.get("temperature", 0.7)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )

        answer = response[0] if isinstance(response, tuple) else response
        outputs = self._parse_outputs(answer, image.width, image.height)
        return VisionResponse.from_outputs(
            *outputs,
            model=request.model,
            task=request.task,
            usage=ModelUsage(output_objects=max(0, len(outputs) - 1)),
        )

    @staticmethod
    def _parse_outputs(text: str, image_width: int, image_height: int):
        boxes = []
        for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
            x1, y1, x2, y2 = [int(group) for group in match.groups()]
            boxes.append(
                Detection(
                    box=BBox(
                        x1=x1 / 1000 * image_width,
                        y1=y1 / 1000 * image_height,
                        x2=x2 / 1000 * image_width,
                        y2=y2 / 1000 * image_height,
                    )
                )
            )
        points = []
        for match in re.finditer(r"<box><(\d+)><(\d+)></box>", text):
            x, y = [int(group) for group in match.groups()]
            points.append(
                PointDetection(
                    point=Point2D(x=x / 1000 * image_width, y=y / 1000 * image_height)
                )
            )
        outputs: list[Any] = [RawTextOutput(text=text)]
        if boxes:
            outputs.append(BoxOutput(boxes=boxes))
        if points:
            outputs.append(PointOutput(points=points))
        return outputs


class SAM21Worker(VisionWorker):
    def __init__(self, model: str, device: str, dtype: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        torch_dtype = getattr(torch, dtype)
        self.device = device
        self.dtype = torch_dtype
        self.processor = Sam2Processor.from_pretrained(model)
        self.model = Sam2Model.from_pretrained(
            model,
            torch_dtype=torch_dtype,
        ).to(device).eval()

    def predict(self, request: VisionRequest) -> VisionResponse:
        import torch

        image = _image_from_input(_first_image(request))
        processor_kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}

        points = []
        labels = []
        boxes = []
        for input_ in request.inputs:
            if input_.type == "point_prompt":
                points.append([input_.point.x, input_.point.y])
                labels.append(0 if input_.label == "negative" else 1)
            elif input_.type == "box_prompt":
                boxes.append(
                    [input_.box.x1, input_.box.y1, input_.box.x2, input_.box.y2]
                )

        if points:
            processor_kwargs["input_points"] = [[points]]
            processor_kwargs["input_labels"] = [[labels]]
        if boxes:
            processor_kwargs["input_boxes"] = [boxes]

        inputs = self.processor(**processor_kwargs).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                multimask_output=request.config.extensions.get(
                    "multimask_output",
                    False,
                ),
            )
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        instances = []
        for index, mask in enumerate(masks.reshape(-1, *masks.shape[-2:])):
            instances.append(
                MaskInstance(
                    id=str(index),
                    mask=MaskRef(
                        data=mask.numpy().astype(bool).tolist(),
                        encoding="bitmap",
                        size=tuple(mask.shape[-2:]),
                    ),
                )
            )
        return VisionResponse.from_outputs(
            MaskOutput(masks=instances, mode="promptable"),
            model=request.model,
            task=request.task,
            usage=ModelUsage(output_objects=len(instances)),
        )


def create_app(worker: VisionWorker):
    from fastapi import FastAPI

    app = FastAPI(title="dspic VM server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/vision")
    def vision(request: dict[str, Any]) -> dict[str, Any]:
        vision_request = VisionRequest.model_validate(request)
        response = worker.predict(vision_request)
        return response.model_dump(mode="json", exclude_none=True)

    return app


def _build_worker(args: argparse.Namespace) -> VisionWorker:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = args.device
    if args.kind == "locateanything":
        return LocateAnythingWorker(args.model, device=device, dtype=args.dtype)
    if args.kind == "sam2.1":
        return SAM21Worker(args.model, device=device, dtype=args.dtype)
    raise ValueError(f"Unknown VM kind: {args.kind}")


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    worker = _build_worker(args)
    app = create_app(worker)
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dspic-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("kind", choices=["locateanything", "sam2.1"])
    serve_parser.add_argument("--model", default=None)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8077)
    serve_parser.add_argument("--gpu", type=int, default=1)
    serve_parser.add_argument("--device", default="cuda")
    serve_parser.add_argument("--dtype", default="bfloat16")

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.model is None:
            args.model = (
                "nvidia/LocateAnything-3B"
                if args.kind == "locateanything"
                else "facebook/sam2.1-hiera-large"
            )
        serve(args)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
