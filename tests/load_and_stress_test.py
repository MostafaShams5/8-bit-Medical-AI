import asyncio
import aiohttp
import time
import statistics

URL = "http://localhost:8000/api/chat"
PAYLOAD = {"question": "ما هي الأسباب الرئيسية للإصابة بمرض السكري؟"}
CONCURRENCY_LEVELS = [10, 50, 100, 150]

async def fire_request(session):
    start = time.perf_counter()
    try:
        async with session.post(URL, json=PAYLOAD, timeout=30) as resp:
            status = resp.status
            await resp.read()
            return time.perf_counter() - start, status
    except Exception:
        return time.perf_counter() - start, 500

async def test_concurrency(level):
    connector = aiohttp.TCPConnector(limit=level)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fire_request(session) for _ in range(level)]
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        latencies = [r[0] for r in results if r[1] == 200]
        errors = len([r for r in results if r[1] != 200])
        
        return {
            "level": level,
            "total_time": total_time,
            "throughput": level / total_time,
            "p50": statistics.median(latencies) if latencies else 0,
            "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0,
            "errors": errors
        }

async def main():
    print("Initiating API Stress Test...")
    for level in CONCURRENCY_LEVELS:
        result = await test_concurrency(level)
        print(f"Concurrency: {result['level']} | RPS: {result['throughput']:.2f} | P50: {result['p50']:.3f}s | Errors: {result['errors']}")

if __name__ == "__main__":
    asyncio.run(main())
