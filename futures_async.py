
import asyncio
import time

async def set_future_result(future, num, delay):
    await asyncio.sleep(delay)
    future.set_result(num ** 2)
    print(f"Set the future result to {future.result()}")
    await asyncio.sleep(delay)
    print("Coroutine finished!")

async def main():
    print('main() started...')

    # Создаем Future-объект
    future = asyncio.Future()

    # Запускаем асинхронную задачу, которая установит значение нашего Future-объекта
    asyncio.create_task(set_future_result(future, 1, 2))

    # Ожидаем не задачу, а наш Future-объект
    await future

    print(f"Results: {future.result()}")
    print('main() finished')


start = time.perf_counter()
asyncio.run(main())
elapsed = time.perf_counter() - start
print(f"{__file__} executed in {elapsed:0.2f} seconds.")
