"""t38_probe_bridge.py — a one-tool MCP server for the image-reach probe (T3.8).

observe() returns a synthetic 512x512 PNG: white background, a large coloured shape, and a 3-digit code in big black
digits. The code and shape are read from the environment (T38_CODE, T38_SHAPE, T38_COLOUR) so the caller knows the
ground truth. Started by codex exec exactly like MIP's bridge (mcp_servers.env.*).
"""

from __future__ import annotations

import os
from io import BytesIO

from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

mcp = FastMCP("probe-env")
CODE = os.environ.get("T38_CODE", "472")
SHAPE = os.environ.get("T38_SHAPE", "circle")
COLOUR = os.environ.get("T38_COLOUR", "red")


def render() -> bytes:
    img = PILImage.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    if SHAPE == "circle":
        d.ellipse([96, 60, 416, 380], fill=COLOUR)
    elif SHAPE == "triangle":
        d.polygon([(256, 50), (60, 380), (452, 380)], fill=COLOUR)
    else:
        d.rectangle([96, 60, 416, 380], fill=COLOUR)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96
        )
    except OSError:
        font = ImageFont.load_default()
    d.text((150, 395), CODE, fill="black", font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool(description="Look through the camera: returns one RGB image.")
def observe() -> list:
    return [Image(data=render(), format="png")]


if __name__ == "__main__":
    mcp.run(transport="stdio")
