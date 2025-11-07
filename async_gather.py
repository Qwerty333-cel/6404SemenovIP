
import asyncio
import time

async def fetch_data(num, delay = 2):
    print(f"Coroutine {num} started...")
    await asyncio.sleep(delay)
    print(f"Coroutine {num} finished")
    return num ** 2

async def main():
    print('main() started...')

    results = await asyncio.gather(fetch_data(1, 2),fetch_data(2, 3),fetch_data(3, 1))

    print(f"Result: {results}")

    print('main() finished')

start = time.perf_counter()
asyncio.run(main())
elapsed = time.perf_counter() - start
print(f"{__file__} executed in {elapsed:0.2f} seconds.")
