
import asyncio
import time
import os
from concurrent.futures import ProcessPoolExecutor

def blocking_sleep(seconds):
    time.sleep(seconds)
    return f"Slept for {seconds} seconds (PID-{os.getpid()})"

async def main():
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_running_loop()
        # Запускаем каждую задачу в отдельном процессе
        tasks = [
            loop.run_in_executor(executor, blocking_sleep, 2),
            loop.run_in_executor(executor, blocking_sleep, 2),
            loop.run_in_executor(executor, blocking_sleep, 2)
        ]
        results = await asyncio.gather(*tasks)

    for result in results:
        print(result)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":  # Важно для ProcessPoolExecutor в Windows!
    asyncio.run(main())
