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

RAW_CSV_PATH = "imu_raw.csv"
PEAK_CSV_PATH = "imu_peaks.csv"

# 최근 몇 초를 기준으로 baseline / threshold를 계산할지
WINDOW_SEC = 5.0

# 그래프에 보여줄 시간 범위
PLOT_SEC = 15.0

# threshold 민감도
# 작을수록 예민, 클수록 둔감
THRESH_K = 4.0

# threshold 최소값
MIN_THRESHOLD_G = 0.02

# peak 사이 최소 간격
MIN_PEAK_DISTANCE_SEC = 0.25

# envelope 부드러움 정도
# 작을수록 부드럽고, 클수록 빠르게 반응
ENVELOPE_ALPHA = 0.2

# threshold 넘은 값만 얼마나 증폭할지
BOOST_GAIN = 3.0

# y축 고정 범위
Y_MAX_G = 0.8


# =========================
# 전역 변수
# =========================

data_queue = queue.Queue()
stop_event = threading.Event()

packet_buffer = bytearray()

# 실시간 그래프용 데이터
times = deque()
acc_mags = deque()
vibs = deque()
envelopes = deque()
enhanceds = deque()
thresholds = deque()

peak_times = deque()
peak_values = deque()

# envelope 상태
prev_envelope = 0.0

# peak 상태
in_peak = False
peak_candidate_time = None
peak_candidate_value = None
peak_candidate_threshold = None
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

    # 가속도 단위: g
    ax = ax_raw / 32768.0 * 16.0
    ay = ay_raw / 32768.0 * 16.0
    az = az_raw / 32768.0 * 16.0

    # 각속도 단위: deg/s
    wx = wx_raw / 32768.0 * 2000.0
    wy = wy_raw / 32768.0 * 2000.0
    wz = wz_raw / 32768.0 * 2000.0

    # 각도 단위: degree
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


def handle_ble_data(data):
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
            data_queue.put(parsed)


# =========================
# BLE 수신
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

        def callback(sender, data):
            handle_ble_data(data)

        await client.start_notify(notify_char, callback)

        print("\n데이터 수신 시작")
        print("그래프 창을 닫으면 종료됨")

        while not stop_event.is_set():
            await asyncio.sleep(0.05)

        await client.stop_notify(notify_char)


def run_ble_thread():
    asyncio.run(ble_main())


# =========================
# Threshold 계산
# =========================

def robust_threshold(values):
    if len(values) < 10:
        return MIN_THRESHOLD_G

    arr = np.array(values)

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad

    threshold = med + THRESH_K * robust_std

    return max(threshold, MIN_THRESHOLD_G)


# =========================
# 샘플 처리 + Peak Detection
# =========================

def process_sample(sample, raw_writer, peak_writer):
    global prev_envelope

    global in_peak
    global peak_candidate_time
    global peak_candidate_value
    global peak_candidate_threshold
    global last_peak_time

    t = sample["time"]
    acc_mag = sample["acc_mag"]

    # 오래된 데이터 삭제
    while times and t - times[0] > PLOT_SEC:
        times.popleft()
        acc_mags.popleft()
        vibs.popleft()
        envelopes.popleft()
        enhanceds.popleft()
        thresholds.popleft()

    while peak_times and t - peak_times[0] > PLOT_SEC:
        peak_times.popleft()
        peak_values.popleft()

    # 최근 WINDOW_SEC 구간 데이터 가져오기
    recent_acc = []
    recent_envelope = []

    for tt, aa, ee in zip(times, acc_mags, envelopes):
        if t - tt <= WINDOW_SEC:
            recent_acc.append(aa)
            recent_envelope.append(ee)

    # baseline 계산
    if len(recent_acc) < 10:
        baseline_acc = acc_mag
    else:
        baseline_acc = np.median(recent_acc)

    # baseline 제거한 진동 성분
    vib = abs(acc_mag - baseline_acc)

    # envelope 적용
    envelope = ENVELOPE_ALPHA * vib + (1 - ENVELOPE_ALPHA) * prev_envelope
    prev_envelope = envelope

    # threshold 계산
    threshold = robust_threshold(recent_envelope)

    # threshold 넘는 부분만 증폭
    if envelope > threshold:
        enhanced = threshold + (envelope - threshold) * BOOST_GAIN
    else:
        enhanced = envelope

    # 저장
    times.append(t)
    acc_mags.append(acc_mag)
    vibs.append(vib)
    envelopes.append(envelope)
    enhanceds.append(enhanced)
    thresholds.append(threshold)

    # peak 판정
    is_above = enhanced > threshold
    is_peak_confirmed = 0

    if is_above:
        if not in_peak:
            if t - last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                in_peak = True
                peak_candidate_time = t
                peak_candidate_value = enhanced
                peak_candidate_threshold = threshold
        else:
            if enhanced > peak_candidate_value:
                peak_candidate_time = t
                peak_candidate_value = enhanced
                peak_candidate_threshold = threshold

    else:
        if in_peak:
            peak_times.append(peak_candidate_time)
            peak_values.append(peak_candidate_value)

            last_peak_time = peak_candidate_time
            is_peak_confirmed = 1

            peak_time_iso = datetime.fromtimestamp(
                peak_candidate_time
            ).isoformat(timespec="milliseconds")

            peak_writer.writerow({
                "peak_time": peak_candidate_time,
                "peak_ts_iso": peak_time_iso,
                "peak_value": peak_candidate_value,
                "threshold": peak_candidate_threshold,
            })

            print(
                f"[PEAK] {peak_time_iso} | "
                f"peak={peak_candidate_value:.4f} g | "
                f"threshold={peak_candidate_threshold:.4f} g"
            )

            in_peak = False
            peak_candidate_time = None
            peak_candidate_value = None
            peak_candidate_threshold = None

    # raw + 처리값 CSV 저장
    raw_writer.writerow({
        "time": sample["time"],
        "ts_iso": sample["ts_iso"],
        "ax": sample["ax"],
        "ay": sample["ay"],
        "az": sample["az"],
        "wx": sample["wx"],
        "wy": sample["wy"],
        "wz": sample["wz"],
        "roll": sample["roll"],
        "pitch": sample["pitch"],
        "yaw": sample["yaw"],
        "acc_mag": sample["acc_mag"],
        "gyro_mag": sample["gyro_mag"],
        "baseline_acc": baseline_acc,
        "vib": vib,
        "envelope": envelope,
        "threshold": threshold,
        "enhanced": enhanced,
        "is_above": int(is_above),
        "is_peak_confirmed": is_peak_confirmed,
    })


# =========================
# 실시간 그래프
# =========================

fig, ax = plt.subplots(figsize=(12, 5))

line_enhanced, = ax.plot([], [], label="enhanced envelope")
line_envelope, = ax.plot([], [], alpha=0.5, label="envelope")
line_threshold, = ax.plot([], [], linestyle="--", label="threshold")
peak_scatter = ax.scatter([], [], marker="o", label="peak")

ax.set_title("WT901BLECL Realtime IMU Peak Detection")
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Vibration magnitude (g)")
ax.set_ylim(0, Y_MAX_G)
ax.grid(True)
ax.legend(loc="upper right")


raw_file = None
peak_file = None
raw_writer = None
peak_writer = None


def update_plot(frame):
    global raw_file
    global peak_file
    global raw_writer
    global peak_writer

    while not data_queue.empty():
        sample = data_queue.get()
        process_sample(sample, raw_writer, peak_writer)

    if raw_file is not None:
        raw_file.flush()

    if peak_file is not None:
        peak_file.flush()

    if not times:
        return line_enhanced, line_envelope, line_threshold, peak_scatter

    now_t = times[-1]

    xs = [tt - now_t for tt in times]

    enhanced_y = list(enhanceds)
    envelope_y = list(envelopes)
    threshold_y = list(thresholds)

    peak_x = [pt - now_t for pt in peak_times]
    peak_y = list(peak_values)

    line_enhanced.set_data(xs, enhanced_y)
    line_envelope.set_data(xs, envelope_y)
    line_threshold.set_data(xs, threshold_y)

    if len(peak_x) > 0:
        peak_scatter.set_offsets(np.column_stack([peak_x, peak_y]))
    else:
        peak_scatter.set_offsets(np.empty((0, 2)))

    ax.set_xlim(-PLOT_SEC, 0)

    return line_enhanced, line_envelope, line_threshold, peak_scatter


def on_close(event):
    stop_event.set()


fig.canvas.mpl_connect("close_event", on_close)


# =========================
# 실행
# =========================

if __name__ == "__main__":
    raw_file = open(RAW_CSV_PATH, "w", newline="", encoding="utf-8")
    peak_file = open(PEAK_CSV_PATH, "w", newline="", encoding="utf-8")

    raw_fieldnames = [
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
        "baseline_acc",
        "vib",
        "envelope",
        "threshold",
        "enhanced",
        "is_above",
        "is_peak_confirmed",
    ]

    peak_fieldnames = [
        "peak_time",
        "peak_ts_iso",
        "peak_value",
        "threshold",
    ]

    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
    peak_writer = csv.DictWriter(peak_file, fieldnames=peak_fieldnames)

    raw_writer.writeheader()
    peak_writer.writeheader()

    ble_thread = threading.Thread(target=run_ble_thread, daemon=True)
    ble_thread.start()

    ani = FuncAnimation(fig, update_plot, interval=50, blit=False)

    try:
        plt.show()
    finally:
        stop_event.set()

        if raw_file is not None:
            raw_file.close()

        if peak_file is not None:
            peak_file.close()

        print("종료됨")