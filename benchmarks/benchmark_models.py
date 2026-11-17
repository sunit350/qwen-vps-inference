import csv
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"

MODELS = [
    "qwen3-0.6b-q4_k_m",
    "qwen3-1.7b-q4_k_m",
    "qwen3-8b-q4_k_m",
    "qwen2.5-coder-1.5b-q4_k_m",
    "deepseek-r1-distill-qwen-1.5b-q4_k_m",
]

PROMPT = (
    "Explain the difference between supervised and "
    "unsupervised learning in two sentences."
)


def benchmark_model(model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)

        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        timings = result.get("timings", {})
        message = result["choices"][0]["message"]

        return {
            "model": model,
            "success": True,
            "total_seconds": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "generation_tokens_per_second": round(
                timings.get("predicted_per_second", 0), 2
            ),
            "prompt_tokens_per_second": round(
                timings.get("prompt_per_second", 0), 2
            ),
            "answer": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "error": "",
        }

    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return {
            "model": model,
            "success": False,
            "total_seconds": round(time.perf_counter() - started, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generation_tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "answer": "",
            "reasoning": "",
            "error": str(exc),
        }


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.csv"

    results = []

    for model in MODELS:
        print(f"\nBenchmarking: {model}", flush=True)
        result = benchmark_model(model)
        results.append(result)

        print(
            f"Success: {result['success']} | "
            f"Time: {result['total_seconds']}s | "
            f"Speed: {result['generation_tokens_per_second']} tok/s",
            flush=True,
        )

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark saved to: {output_file}")


if __name__ == "__main__":
    main()