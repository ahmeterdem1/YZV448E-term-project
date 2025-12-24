#!/usr/bin/env python3
"""
Stress Test Script for PII Cleaner
Designed to force computation limits, hit rate limits, and fill queues.
"""

import requests
import time
import random
import string
import concurrent.futures
from typing import List, Dict

# --- Configuration ---
BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 50  # Total requests to send (exceeds default rate limit of 20)
CONCURRENT_THREADS = 10  # How many requests to send at once
TEXT_LENGTH = 9500  # Close to MAX_TEXT_LENGTH (10,000)
VARY_LENGTH = True  # If True, lengths vary from 1000 to TEXT_LENGTH

# Fake PII data to inject
PII_TEMPLATES = [
    "My email is test.{i}@example.com",
    "Call me at 555-{i:04d}",
    "SSN: {i:03d}-00-{i:04d}",
    "Credit Card: 4000-1234-5678-{i:04d}",
    "I live at {i} Main St, New York, NY 10001"
]


def generate_heavy_payload(index: int) -> str:
    """Generates a large text block filled with PII and random junk."""
    base_pii = " ".join([t.format(i=index + k) for k, t in enumerate(PII_TEMPLATES)])

    target_len = TEXT_LENGTH
    if VARY_LENGTH:
        target_len = random.randint(1000, TEXT_LENGTH)

    # Fill remaining space with random words to force tokenizer work
    padding_len = max(0, target_len - len(base_pii))
    padding = ''.join(random.choices(string.ascii_letters + " ", k=padding_len))

    return base_pii + " " + padding


def send_stress_request(index: int) -> Dict:
    """Sends a single heavy request and returns the result/status."""
    text_content = generate_heavy_payload(index)
    payload = {"text_content": text_content}

    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/process-text",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10  # fast timeout to detect lags
        )
        elapsed = time.time() - start_time
        return {
            "index": index,
            "status": response.status_code,
            "elapsed": elapsed,
            "size": len(text_content),
            "response": response.text[:100]  # Log first 100 chars
        }
    except Exception as e:
        return {
            "index": index,
            "status": "ERROR",
            "elapsed": time.time() - start_time,
            "size": len(text_content),
            "response": str(e)
        }


def run_stress_test():
    print(f"🔥 STARTING STRESS TEST")
    print(f"🎯 Target: {BASE_URL}")
    print(f"📦 Requests: {NUM_REQUESTS} | Threads: {CONCURRENT_THREADS}")
    print(f"📄 Avg Payload Size: ~{TEXT_LENGTH} chars")
    print("=" * 60)

    stats = {201: 0, 429: 0, 503: 0, "ERROR": 0, "OTHER": 0}
    task_ids = []

    # 1. Burst Processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        futures = [executor.submit(send_stress_request, i) for i in range(NUM_REQUESTS)]

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            status = res['status']

            # Update stats
            if status in stats:
                stats[status] += 1
            else:
                stats["OTHER"] += 1

            # Log based on status
            symbol = "✅"
            if status == 429:
                symbol = "⛔"  # Rate Limited
            elif status == 503:
                symbol = "🔥"  # Queue Full / Model Overload
            elif status == 413:
                symbol = "📏"  # Too Large
            elif status == "ERROR":
                symbol = "❌"

            print(f"{symbol} Req #{res['index']:02d} | Status: {status} | "
                  f"Size: {res['size']} | Time: {res['elapsed']:.2f}s")

            # Capture success IDs for checking later
            if status == 201:
                try:
                    import json
                    data = json.loads(res['response'] + "}")  # Hacky fix if truncated, or better:
                    # In real flow, we'd parse properly. Here we just skipped parsing in send_stress_request
                    # Let's trust the print output generally, or fetch from queue list later.
                    pass
                except:
                    pass

    print("=" * 60)
    print("📊 STRESS TEST RESULTS")
    print(f"✅ Successful (201): {stats[201]}")
    print(f"⛔ Rate Limited (429): {stats[429]} (Expected if > 20/min)")
    print(f"🔥 Queue Full (503):   {stats[503]}")
    print(f"❌ Connection Errors:  {stats['ERROR']}")
    print("=" * 60)

    # 2. Queue Drain Monitor
    print("👀 Monitoring Queue Drain...")
    while True:
        try:
            r = requests.get(f"{BASE_URL}/api/v1/queue")
            queue = r.json()
            remaining = len(queue)

            print(f"   Queue Size: {remaining} items pending...")

            if remaining == 0:
                print("✅ Queue drained.")
                break

            time.sleep(2)
        except Exception as e:
            print(f"Error checking queue: {e}")
            break


if __name__ == "__main__":
    # Optional: Verify health first
    try:
        if requests.get(f"{BASE_URL}/health").status_code != 200:
            print("❌ Server is down! Start docker containers first.")
            exit(1)
    except:
        print("❌ Server is unreachable! Start docker containers first.")
        exit(1)

    run_stress_test()