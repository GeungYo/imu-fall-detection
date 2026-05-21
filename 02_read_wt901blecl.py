import asyncio
import csv
import math
import time
from datetime import datetime
from bleak import BleakClient

# 01_scan_ble.py에서 찾은 address로 바꾸기
DEVICE_ADDRESS = "DD:D6:0F:01:23:A5"

CSV_PATH = "imu_data.csv"

buffer = bytearray()


def to_int16(low, high):
    value = (high << 8) | low
    if value >= 32768:
        value -= 65536
    return value


def parse_packet(packet):
    """
    WT901BLECL 기본 패킷:
    0x55 0x61 + 18 bytes
    ax, ay, az, wx, wy, wz, roll, pitch, yaw
    """

    if len(packet) != 20:
        return None

    if packet[0] != 0x55 or packet[1] != 0x61:
        return None

    ax_raw = to_int16(packet[2], packet[3])
    ay_raw = to_int16(packet[4], packet[5])
    az_raw = to_int16(packet[6], packet[7])

    wx_raw = to_int16(packet[8], packet[9])
    wy_raw = to_int16(packet[10], packet[11])
    wz_raw = to_int16(packet[12], packet[13])

    roll_raw = to_int16(packet[14], packet[15])
    pitch_raw = to_int16(packet[16], packet[17])
    yaw_raw = to_int16(packet[18], packet[19])

    # 공식 변환식 기준
    ax = ax_raw / 32768.0 * 16.0
    ay = ay_raw / 32768.0 * 16.0
    az = az_raw / 32768.0 * 16.0

    wx = wx_raw / 32768.0 * 2000.0
    wy = wy_raw / 32768.0 * 2000.0
    wz = wz_raw / 32768.0 * 2000.0

    roll = roll_raw / 32768.0 * 180.0
    pitch = pitch_raw / 32768.0 * 180.0
    yaw = yaw_raw / 32768.0 * 180.0

    acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = math.sqrt(wx * wx + wy * wy + wz * wz)

    return {
        "time": time.time(),
        "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
        "ax": ax,
        "ay": ay,
        "az": az,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "acc_mag": acc_mag,
        "gyro_mag": gyro_mag,
    }


def handle_packet(data, writer):
    global buffer

    buffer.extend(data)

    while len(buffer) >= 20:
        # 0x55 0x61 시작 위치 찾기
        start = -1
        for i in range(len(buffer) - 1):
            if buffer[i] == 0x55 and buffer[i + 1] == 0x61:
                start = i
                break

        if start == -1:
            buffer.clear()
            return

        if start > 0:
            del buffer[:start]

        if len(buffer) < 20:
            return

        packet = buffer[:20]
        del buffer[:20]

        parsed = parse_packet(packet)

        if parsed is not None:
            writer.writerow(parsed)

            print(
                f"acc_mag={parsed['acc_mag']:.3f} g | "
                f"gyro_mag={parsed['gyro_mag']:.2f} deg/s | "
                f"ax={parsed['ax']:.3f}, ay={parsed['ay']:.3f}, az={parsed['az']:.3f}"
            )


async def main():
    print("센서 연결 중...")

    async with BleakClient(DEVICE_ADDRESS) as client:
        print("연결 성공")

        notify_char = None

        print("\n사용 가능한 서비스/캐릭터리스틱:")
        for service in client.services:
            print("[Service]", service.uuid)

            for char in service.characteristics:
                print("  [Char]", char.uuid, char.properties)

                if "notify" in char.properties and notify_char is None:
                    notify_char = char.uuid

        if notify_char is None:
            print("notify 가능한 characteristic을 찾지 못했어.")
            return

        print("\nnotify characteristic:", notify_char)

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "time",
                "ts_iso",
                "ax",
                "ay",
                "az",
                "wx",
                "wy",
                "wz",
                "roll",
                "pitch",
                "yaw",
                "acc_mag",
                "gyro_mag",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            def callback(sender, data):
                handle_packet(data, writer)

            await client.start_notify(notify_char, callback)

            print("\n데이터 수신 시작")
            print("종료하려면 Ctrl + C")

            while True:
                await asyncio.sleep(1)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n종료됨")