"""Image input check against a running service (temperature 0): synthetic images whose content is known.

    python3 tools/bench/vision_check.py [--base http://127.0.0.1:8000] [--model NAME]

Each case draws a code word and coloured shapes, asks for them and checks the answer; then two images in one
request, a follow-up turn on the same image (prefix cache) and a text-only request afterwards. One JSON line per
case; needs Pillow.
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (20, 160, 60), "yellow": (240, 200, 0)}


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def picture(width, height, word, shape_color, background=(255, 255, 255)):
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    s = min(width, height)
    draw.ellipse([width * 0.08, height * 0.1, width * 0.08 + s * 0.35, height * 0.1 + s * 0.35], fill=COLORS[shape_color])
    draw.text((width * 0.5, height * 0.4), word, fill=(0, 0, 0), font=font(max(16, s // 8)))
    return image


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def chat(args, messages, max_tokens=80):
    body = {"model": args.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    t0 = time.perf_counter()
    request = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            r = json.loads(response.read())
    except urllib.error.HTTPError as error:
        return None, {"http": error.code, "body": error.read().decode()[:300]}, time.perf_counter() - t0
    return r["choices"][0]["message"]["content"], r["usage"], time.perf_counter() - t0


def user(text, *images):
    parts = [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
    return {"role": "user", "content": parts + [{"type": "text", "text": text}]}


QUESTION = ("What word is written in the image, and what colour is the round shape? "
            "Answer exactly as: word=<WORD>, color=<colour name>")


def report(case, answer, usage, seconds, expect):
    ok = answer is not None and all(e.lower() in answer.lower() for e in expect)
    print(json.dumps({"case": case, "ok": ok, "seconds": round(seconds, 2), "usage": usage,
                      "answer": (answer or "")[:160]}, ensure_ascii=False), flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "qwen3.8-27b-fp8-mtp"))
    args = parser.parse_args()
    results = []
    for (w, h), word, color in (((640, 480), "MANGO", "red"), ((1920, 1080), "VIOLIN", "blue"),
                                ((300, 300), "ORBIT", "green"), ((1024, 1024), "CACTUS", "yellow"),
                                # new sizes that land in buckets already used above (fast with VISION_BUCKETS)
                                ((1000, 700), "LEMON", "red"), ((990, 730), "CANDLE", "blue"),
                                ((1100, 650), "ROCKET", "green"), ((500, 420), "PEPPER", "yellow")):
        answer, usage, seconds = chat(args, [user(QUESTION, picture(w, h, word, color))])
        results.append(report(f"single {w}x{h}", answer, usage, seconds, [word, color]))
    a, b = picture(800, 600, "PIANO", "red"), picture(600, 800, "TIGER", "blue")
    answer, usage, seconds = chat(args, [user("Two images. What word is written in each? Answer as: first=<W>, "
                                              "second=<W>", a, b)])
    results.append(report("two images", answer, usage, seconds, ["PIANO", "TIGER"]))
    image = picture(1280, 720, "HARBOR", "green")
    first, usage, seconds = chat(args, [user(QUESTION, image)])
    results.append(report("turn 1", first, usage, seconds, ["HARBOR", "green"]))
    follow = [user(QUESTION, image), {"role": "assistant", "content": first or ""},
              {"role": "user", "content": "Now spell that word backwards, letters only."}]
    answer, usage, seconds = chat(args, follow)
    results.append(report("turn 2 (same image, prefix cache)", answer, usage, seconds, ["ROBRAH"]))
    answer, usage, seconds = chat(args, [{"role": "user", "content": "What is 17 * 23? Answer with the number."}], 20)
    results.append(report("text only afterwards", answer, usage, seconds, ["391"]))
    print(json.dumps({"passed": sum(results), "total": len(results)}), flush=True)


if __name__ == "__main__":
    main()
