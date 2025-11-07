
import asyncio

def callback():
    print("This function was called after 5 seconds!")

async def main():
    loop = asyncio.get_event_loop()
    loop.call_later(5, callback)  # Вызовем задачу через 5 секунд
    await asyncio.sleep(6)  # Чтобы loop успел выполнить задачу до закрытия

if __name__ == "__main__":
  asyncio.run(main())
