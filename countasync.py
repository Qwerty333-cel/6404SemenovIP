
import asyncio

async def count(delay = 2):
    print("One")
    await asyncio.sleep(delay)
    print("Two")

async def main():
    print('main() started...')
    coro = count(2)
    await coro
    print('main() finished')

asyncio.run(main())
