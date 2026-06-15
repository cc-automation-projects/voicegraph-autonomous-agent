import asyncio
import time

import httpx

BASE_URL = "http://propensity-inference:8000"
NUM_REQUESTS = 100
CONCURRENCY = 10


async def single_request(client: httpx.AsyncClient) -> float:
    start = time.monotonic()
    try:
        response = await client.post(
            f"{BASE_URL}/predict",
            json={
                "campaign_id": "benchmark",
                "users": [
                    {
                        "user_id": "00000000-0000-0000-0000-000000000001",
                        "phone_hash": "a" * 64,
                        "days_since_contact": 1,
                    }
                ],
            },
        )
        await response.aread()
    except Exception:
        pass
    return (time.monotonic() - start) * 1000


async def run_benchmark():
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [single_request(client) for _ in range(NUM_REQUESTS)]
        latencies = await asyncio.gather(*tasks)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"Benchmark: {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")
    print(f"P50 latency: {p50:.2f}ms")
    print(f"P99 latency: {p99:.2f}ms")
    print(f"Mean latency: {sum(latencies) / len(latencies):.2f}ms")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
