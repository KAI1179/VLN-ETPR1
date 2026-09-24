import os
import time
import litellm

print("litellm", litellm.__version__ if hasattr(litellm, "__version__") else "imported")
tests = []
if os.getenv("DASHSCOPE_API_KEY"):
    tests.append(("dashscope/qwen-plus", {}))
if os.getenv("OPENAI_API_KEY"):
    tests.append(("gpt-4o-mini", {}))
if os.getenv("ANTHROPIC_API_KEY"):
    tests.append(("claude-3-5-haiku-latest", {}))
for m, kw in tests:
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
    except Exception as ex:
        print(m, "FAIL", type(ex).__name__, str(ex)[:300])
if not tests:
    print("no provider keys found")
