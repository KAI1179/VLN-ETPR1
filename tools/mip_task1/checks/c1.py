"""C1: provider reachability through litellm, plus one image (vision) probe.

Env: provider keys as usual; DASHSCOPE_API_BASE (default cn compatible-mode),
VISION_MODEL (default per provider). Never prints keys.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time


def png_64x64_red() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    import litellm

    dash_base = os.environ.get(
        "DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    text_tests: list[tuple[str, dict]] = []
    vision_tests: list[tuple[str, dict]] = []
    if os.getenv("DASHSCOPE_API_KEY"):
        kw = {"api_base": dash_base, "api_key": os.environ["DASHSCOPE_API_KEY"]}
        text_tests.append(("openai/qwen-plus", kw))
        vision_tests.append((
            os.environ.get("VISION_MODEL", "openai/qwen3-vl-plus"),
            kw,
        ))
    if os.getenv("OPENAI_API_KEY"):
        text_tests.append(("gpt-4o-mini", {}))
        vision_tests.append(("gpt-4o-mini", {}))
    if os.getenv("ANTHROPIC_API_KEY"):
        text_tests.append(("claude-3-5-haiku-latest", {}))
        vision_tests.append(("claude-3-5-haiku-latest", {}))
    if not text_tests:
        print("no provider keys found")
        return 2
    for m, kw in text_tests:
        t = time.time()
        try:
            r = litellm.completion(
                model=m,
                messages=[{"role": "user", "content": "reply with the single word ok"}],
                max_tokens=5,
                **kw,
            )
            print(
                m,
                "OK",
                repr(r.choices[0].message.content),
                f"{time.time() - t:.1f}s",
                r.usage,
            )
        except Exception as ex:  # noqa: BLE001
            print(m, "FAIL", type(ex).__name__, str(ex)[:300])
    img = png_64x64_red()
    for m, kw in vision_tests:
        t = time.time()
        try:
            r = litellm.completion(
                model=m,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What colour is this image? One word.",
                            },
                            {"type": "image_url", "image_url": {"url": img}},
                        ],
                    }
                ],
                max_tokens=10,
                **kw,
            )
            print(
                "VISION",
                m,
                "OK",
                repr(r.choices[0].message.content),
                f"{time.time() - t:.1f}s",
                r.usage,
            )
        except Exception as ex:  # noqa: BLE001
            print("VISION", m, "FAIL", type(ex).__name__, str(ex)[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
