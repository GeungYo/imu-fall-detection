import asyncio
from bleak import BleakScanner

async def main():
    print("BLE 장치 검색 중...")

    devices = await BleakScanner.discover(timeout=8)

    for d in devices:
        print("--------------------")
        print("name   :", d.name)
        print("address:", d.address)

asyncio.run(main())

'''
name   : WT901BLE68
address: DD:D6:0F:01:23:A5
'''