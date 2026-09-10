"""Resolution Selector with the 4:5 and 5:4 social ratios added.

Same math and widgets as the core ``ResolutionSelector``
(``comfy_extras/nodes_resolution.py``), but the aspect ratio list also
carries the 4:5 / 5:4 pair that the core node is missing.
"""

import math
from enum import Enum

from comfy_api.latest import io


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_V = "2:3 (Portrait Photo)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    STANDARD_H = "4:3 (Standard)"
    SOCIAL_V = "4:5 (Portrait Social)"
    SOCIAL_H = "5:4 (Social)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.SOCIAL_V: (4, 5),
    AspectRatio.SOCIAL_H: (5, 4),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
}


class MDResolutionSelector(io.ComfyNode):
    """Calculate width and height from aspect ratio and megapixel target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MDResolutionSelector",
            display_name="Resolution Selector (MD)",
            category="MDPack/utilities",
            description="Calculate width and height from aspect ratio and megapixel target. "
                        "Same as the core Resolution Selector, plus the 4:5 and 5:4 ratios.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="The aspect ratio for the output dimensions.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target total megapixels. 1.0 MP = 1024x1024 for square.",
                ),
                io.ResolutionPreview.Input(
                    "preview",
                    tooltip="Live preview of the calculated output resolution.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=8,
                    min=8,
                    max=128,
                    step=4,
                    tooltip="Nearest multiple of the result to set the selected resolution to.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Calculated width in pixels, rounded to the selected multiple."),
                io.Int.Output("height", tooltip="Calculated height in pixels, rounded to the selected multiple."),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, megapixels: float, multiple: int, preview=None) -> io.NodeOutput:
        width_ratio, height_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (width_ratio * height_ratio))
        width = round(width_ratio * scale / multiple) * multiple
        height = round(height_ratio * scale / multiple) * multiple
        return io.NodeOutput(width, height)


# The pack registers V1-style, so the V3 schema is finalized here instead of
# through comfy_entrypoint (category/description would stay unset otherwise).
MDResolutionSelector.GET_SCHEMA()

NODE_CLASS_MAPPINGS = {
    "MDResolutionSelector": MDResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MDResolutionSelector": "Resolution Selector (MD)",
}
