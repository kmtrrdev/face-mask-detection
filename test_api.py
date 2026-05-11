"""
🧪 test_api.py — Automated API Testing for Face Mask Detection
Usage:
    python test_api.py                  # uses http://localhost:8000
    python test_api.py --url http://... # custom base URL
"""

import argparse
import io
import json
import sys
import time
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = "/predict"
HEALTH_ENDPOINT  = "/health"

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results: list[dict] = []


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_dummy_image(color: tuple = (128, 128, 128), size: tuple = (224, 224)) -> bytes:
    """Create a plain-color JPEG image in memory."""
    img  = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf  = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def post_image(url: str, image_bytes: bytes, filename: str = "test.jpg") -> tuple[int, dict]:
    """
    Send a multipart POST request with the image.
    Returns (status_code, response_json).
    """
    boundary  = b"----TestBoundary12345"
    body      = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
        + image_bytes
        + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def record(test_name: str, passed: bool, detail: str = ""):
    icon = PASS if passed else FAIL
    msg  = f"  {icon}  {test_name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append({"name": test_name, "passed": passed, "detail": detail})


# ─── Test cases ──────────────────────────────────────────────────────────────

def test_health(base_url: str):
    print("\n── Health Check ─────────────────────────────────────────────")
    url = base_url + HEALTH_ENDPOINT
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data   = json.loads(resp.read())
            passed = resp.status == 200 and data.get("status") == "ok"
            record("GET /health returns 200 with status=ok", passed, str(data))
    except Exception as exc:
        record("GET /health returns 200 with status=ok", False, str(exc))


def test_predict_valid_image(base_url: str):
    print("\n── /predict — Valid Inputs ──────────────────────────────────")
    url = base_url + PREDICT_ENDPOINT

    # Test with a generic grey image (model may return any class — we just check structure)
    img_bytes = make_dummy_image(color=(128, 128, 128))
    status, data = post_image(url, img_bytes, "grey_face.jpg")

    structure_ok = (
        status == 200
        and "status"        in data
        and "action"        in data
        and "class"         in data
        and "confidence"    in data
        and "probabilities" in data
    )
    record("Valid JPEG → 200 with all required fields", structure_ok, str(data))

    if structure_ok:
        # confidence should be between 0 and 1
        conf_ok = 0.0 <= data["confidence"] <= 1.0
        record("Confidence is in [0, 1]", conf_ok, f"confidence={data['confidence']}")

        # probabilities should sum to ~1
        probs = list(data["probabilities"].values())
        sum_ok = abs(sum(probs) - 1.0) < 0.01
        record("Probabilities sum to ~1", sum_ok, f"sum={sum(probs):.4f}")

        # action must be a meaningful string
        action_ok = data["action"] in ("Allow entry", "Deny entry")
        record("Action is 'Allow entry' or 'Deny entry'", action_ok, f"action={data['action']}")

        # status must match class
        status_ok = (
            (data["class"] == "WithMask"    and data["status"] == "mask_on")  or
            (data["class"] == "WithoutMask" and data["status"] == "mask_off")
        )
        record("Status is consistent with class", status_ok)


def test_predict_invalid_inputs(base_url: str):
    print("\n── /predict — Invalid Inputs ────────────────────────────────")
    url = base_url + PREDICT_ENDPOINT

    # 1. Non-image file (plain text)
    txt_bytes = b"this is not an image"
    boundary  = b"----TestBoundary99"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="bad.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        + txt_bytes
        + b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            record("Non-image file → 4xx error", resp.status >= 400, f"got {resp.status}")
    except urllib.error.HTTPError as exc:
        record("Non-image file → 4xx error", exc.code in (400, 415, 422), f"got {exc.code}")
    except Exception as exc:
        record("Non-image file → 4xx error", False, str(exc))

    # 2. No file at all → 422 Unprocessable Entity
    req2 = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req2, timeout=10) as resp:
            record("Empty body → 422 error", resp.status == 422, f"got {resp.status}")
    except urllib.error.HTTPError as exc:
        record("Empty body → 422 error", exc.code == 422, f"got {exc.code}")
    except Exception as exc:
        record("Empty body → 422 error", False, str(exc))


def test_predict_image_variants(base_url: str):
    print("\n── /predict — Image Variants ────────────────────────────────")
    url = base_url + PREDICT_ENDPOINT

    # Small image (32×32)
    img_bytes  = make_dummy_image(size=(32, 32))
    status, _  = post_image(url, img_bytes, "small.jpg")
    record("32×32 tiny image → 200", status == 200, f"got {status}")

    # Large image (1920×1080)
    img_bytes  = make_dummy_image(size=(1920, 1080))
    status, _  = post_image(url, img_bytes, "large.jpg")
    record("1920×1080 large image → 200", status == 200, f"got {status}")

    # PNG format
    img = Image.fromarray(np.full((224, 224, 3), 200, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    status, _ = post_image(url, buf.getvalue(), "test.png")
    record("PNG image → 200", status == 200, f"got {status}")


def test_response_time(base_url: str):
    print("\n── Performance ──────────────────────────────────────────────")
    url       = base_url + PREDICT_ENDPOINT
    img_bytes = make_dummy_image()

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        post_image(url, img_bytes)
        times.append(time.perf_counter() - t0)

    avg_ms = (sum(times) / len(times)) * 1000
    fast   = avg_ms < 2000  # 2 seconds threshold
    record(
        "Average inference < 2 000 ms",
        fast,
        f"avg={avg_ms:.0f} ms  (3 requests)",
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Mask API test suite")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="API base URL")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n🧪 Running test suite against  {base_url}")
    print("=" * 60)

    # Wait for server to be ready (up to 30 s)
    print("\nWaiting for API to be reachable…", end="", flush=True)
    for _ in range(30):
        try:
            urllib.request.urlopen(base_url + HEALTH_ENDPOINT, timeout=2)
            print(" ready ✅")
            break
        except Exception:
            time.sleep(1)
            print(".", end="", flush=True)
    else:
        print("\n❌ API did not respond in time. Is the server running?")
        sys.exit(1)

    test_health(base_url)
    test_predict_valid_image(base_url)
    test_predict_invalid_inputs(base_url)
    test_predict_image_variants(base_url)
    test_response_time(base_url)

    # ── Summary ─────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"{'🎉 ALL TESTS PASSED' if failed == 0 else '⚠️  SOME TESTS FAILED'}")
    print(f"   Passed : {passed} / {total}")
    if failed:
        print(f"   Failed : {failed} / {total}")
        for r in results:
            if not r["passed"]:
                print(f"     • {r['name']}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
