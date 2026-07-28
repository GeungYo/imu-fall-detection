import asyncio
import csv
import math
import time
import threading
import queue
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bleak import BleakClient



# =========================
# 설정값
# =========================

DEVICE_ADDRESS = "DD:D6:0F:01:23:A5"

CSV_PATH = "imu_realtime_peak.csv"

# threshold 계산에 사용할 최근 윈도우 크기
WINDOW_SEC = 5.0

# 그래프에 보여줄 시간 범위
PLOT_SEC = 15.0

# threshold 민감도
# 작게 하면 peak가 잘 잡히고, 크게 하면 강한 충격만 잡힘
THRESH_K = 5.0

# threshold가 너무 작아지는 것을 방지하는 최소값
# 단위: g
MIN_THRESHOLD_G = 0.03

# peak 사이 최소 간격
MIN_PEAK_DISTANCE_SEC = 0.25

# 그래프 y축 고정
Y_MAX_G = 0.6


# =========================
# 전역 변수
# =========================

data_queue = queue.Queue()
stop_event = threading.Event()

packet_buffer = bytearray()

# plot용 저장 공간
times = deque()
acc_mags = deque()
vibs = deque()
thresholds = deque()

peak_times = deque()
peak_values = deque()

# peak detection 상태
in_peak = False
peak_candidate_time = None
peak_candidate_value = None
last_peak_time = -999.0


# =========================
# WT901BLECL 패킷 파싱
# =========================

def to_int16(low, high):
    value = (high << 8) | low
    if value >= 32768:
        value -= 65536
    return value


def parse_packet(packet):
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

    # 변환
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

    now = time.time()

    return {
        "time": now,
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


def handle_ble_data(data, writer):
    global packet_buffer

    packet_buffer.extend(data)

    while len(packet_buffer) >= 20:
        start = -1

        for i in range(len(packet_buffer) - 1):
            if packet_buffer[i] == 0x55 and packet_buffer[i + 1] == 0x61:
                start = i
                break

        if start == -1:
            packet_buffer.clear()
            return

        if start > 0:
            del packet_buffer[:start]

        if len(packet_buffer) < 20:
            return

        packet = packet_buffer[:20]
        del packet_buffer[:20]

        parsed = parse_packet(packet)

        if parsed is not None:
            writer.writerow(parsed)
            data_queue.put(parsed)


# =========================
# BLE 수신 스레드
# =========================

async def ble_main():
    print("센서 연결 중...")

    async with BleakClient(DEVICE_ADDRESS) as client:
        print("연결 성공")

        notify_char = None

        print("\n사용 가능한 characteristic:")
        for service in client.services:
            for char in service.characteristics:
                print(char.uuid, char.properties)

                if "notify" in char.properties and notify_char is None:
                    notify_char = char.uuid

        if notify_char is None:
            print("notify characteristic을 찾지 못했어.")
            return

        print("\n사용할 notify characteristic:", notify_char)

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
                handle_ble_data(data, writer)

            await client.start_notify(notify_char, callback)

            print("\n데이터 수신 시작")
            print("그래프 창을 닫으면 종료돼.")

            while not stop_event.is_set():
                await asyncio.sleep(0.05)

            await client.stop_notify(notify_char)


def run_ble_thread():
    asyncio.run(ble_main())


# =========================
# Threshold / Peak Detection
# =========================

def robust_threshold(values):
    """
    최근 window의 vib 값으로 threshold 계산.
    median + K * MAD 방식.
    """
    if len(values) < 10:
        return None

    arr = np.array(values)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad
    th = med + THRESH_K * robust_std

    return max(th, MIN_THRESHOLD_G)


def process_sample(sample):
    global in_peak
    global peak_candidate_time
    global peak_candidate_value
    global last_peak_time

    t = sample["time"]
    acc_mag = sample["acc_mag"]

    # 오래된 데이터 삭제
    while times and t - times[0] > PLOT_SEC:
        times.popleft()
        acc_mags.popleft()
        vibs.popleft()
        thresholds.popleft()

    while peak_times and t - peak_times[0] > PLOT_SEC:
        peak_times.popleft()
        peak_values.popleft()

    # 최근 WINDOW_SEC 구간 추출
    recent_acc = []
    recent_vib = []

    for tt, aa, vv in zip(times, acc_mags, vibs):
        if t - tt <= WINDOW_SEC:
            recent_acc.append(aa)
            recent_vib.append(vv)

    # acc 기준값
    if len(recent_acc) < 10:
        baseline_acc = acc_mag
    else:
        baseline_acc = np.median(recent_acc)

    # 진동 크기
    vib = abs(acc_mag - baseline_acc)

    # threshold 계산
    th = robust_threshold(recent_vib)

    if th is None:
        th = MIN_THRESHOLD_G

    times.append(t)
    acc_mags.append(acc_mag)
    vibs.append(vib)
    thresholds.append(th)

    # peak 판정
    is_above = vib > th

    if is_above:
        if not in_peak:
            if t - last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                in_peak = True
                peak_candidate_time = t
                peak_candidate_value = vib
        else:
            if vib > peak_candidate_value:
                peak_candidate_time = t
                peak_candidate_value = vib

    else:
        if in_peak:
            peak_times.append(peak_candidate_time)
            peak_values.append(peak_candidate_value)

            last_peak_time = peak_candidate_time

            print(
                f"[PEAK] time={datetime.now().strftime('%H:%M:%S.%f')[:-3]} | "
                f"value={peak_candidate_value:.4f} g | "
                f"threshold={th:.4f} g"
            )

            in_peak = False
            peak_candidate_time = None
            peak_candidate_value = None


# =========================
# 실시간 그래프
# =========================

fig, ax = plt.subplots(figsize=(12, 5))

line_vib, = ax.plot([], [], label="vib = |acc_mag - baseline|")
line_th, = ax.plot([], [], linestyle="--", label="threshold")
peak_scatter = ax.scatter([], [], marker="o", label="peak")

ax.set_title("WT901BLECL Realtime Vibration Peak Detection")
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Vibration magnitude (g)")
ax.set_ylim(0, Y_MAX_G)
ax.grid(True)
ax.legend(loc="upper right")


def update_plot(frame):
    # queue에 쌓인 새 데이터 처리
    while not data_queue.empty():
        sample = data_queue.get()
        process_sample(sample)

    if not times:
        return line_vib, line_th, peak_scatter

    t0 = times[-1]

    xs = [tt - t0 for tt in times]
    ys = list(vibs)
    ths = list(thresholds)

    px = [pt - t0 for pt in peak_times]
    py = list(peak_values)

    line_vib.set_data(xs, ys)
    line_th.set_data(xs, ths)

    if len(px) > 0:
        peak_scatter.set_offsets(np.column_stack([px, py]))
    else:
        peak_scatter.set_offsets(np.empty((0, 2)))

    ax.set_xlim(-PLOT_SEC, 0)

    return line_vib, line_th, peak_scatter


def on_close(event):
    stop_event.set()


fig.canvas.mpl_connect("close_event", on_close)


# =========================
# 실행
# =========================

if __name__ == "__main__":
    ble_thread = threading.Thread(target=run_ble_thread, daemon=True)
    ble_thread.start()

    ani = FuncAnimation(fig, update_plot, interval=50, blit=False)
    plt.show()

    stop_event.set()
    print("종료됨")