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

CSV_PATH = "imu_dynamic_threshold.csv"

# 기준값 / 노이즈 계산에 사용할 최근 윈도우 크기
WINDOW_SEC = 5.0

# 그래프에 보여줄 시간 범위
PLOT_SEC = 15.0

# peak 사이 최소 간격
MIN_PEAK_DISTANCE_SEC = 0.25


# =========================
# 증폭 설정값
# =========================

# envelope 부드러움 정도
# 클수록 빠르게 반응, 작을수록 부드러움
ENVELOPE_ALPHA = 0.3

# score 증폭 강도
SCORE_BOOST_GAIN = 16.0

# noise_level이 너무 작아지는 것 방지
NOISE_EPS = 0.0015

# 너무 작은 실제 진동은 peak로 보지 않기
MIN_SIGNAL_G = 0.002


# =========================
# 동적 threshold 설정값
# =========================

# 최근 boosted_score 분포를 보고 threshold를 계산할 때 민감도
# 작게 하면 예민, 크게 하면 둔감
DYNAMIC_THRESH_K = 5.0

# threshold가 너무 낮아지는 것 방지
MIN_SCORE_THRESHOLD = 15.0

# 큰 충격 때문에 threshold가 너무 높아지는 것 방지
MAX_SCORE_THRESHOLD = 60.0

# 그래프 y축 고정
Y_MAX_SCORE = 80


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
envelopes = deque()
noise_levels = deque()
scores = deque()
boosted_scores = deque()
dynamic_thresholds = deque()

peak_times = deque()
peak_values = deque()

# envelope 상태
prev_envelope = 0.0

# peak detection 상태
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

        def callback(sender, data):
            handle_ble_data(data)

        await client.start_notify(notify_char, callback)

        print("\n데이터 수신 시작")
        print("그래프 창을 닫으면 종료돼.")

        while not stop_event.is_set():
            await asyncio.sleep(0.05)

        await client.stop_notify(notify_char)


def run_ble_thread():
    asyncio.run(ble_main())


# =========================
# Noise 계산
# =========================

def robust_noise(values):
    """
    최근 window의 envelope 값으로 평소 노이즈 수준 계산
    median + MAD 방식
    """

    if len(values) < 10:
        return NOISE_EPS

    arr = np.array(values)

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad
    noise_level = med + robust_std

    return max(noise_level, NOISE_EPS)


# =========================
# 동적 threshold 계산
# =========================

def dynamic_score_threshold(values):
    """
    최근 boosted_score 분포를 보고 threshold 자동 계산.
    큰 peak가 threshold 계산을 망치지 않도록 상위 10%는 제외.
    """

    if len(values) < 10:
        return MIN_SCORE_THRESHOLD

    arr = np.array(values)

    upper = np.percentile(arr, 90)
    arr = arr[arr <= upper]

    if len(arr) < 10:
        return MIN_SCORE_THRESHOLD

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad

    threshold = med + DYNAMIC_THRESH_K * robust_std

    threshold = max(threshold, MIN_SCORE_THRESHOLD)
    threshold = min(threshold, MAX_SCORE_THRESHOLD)

    return threshold


# =========================
# 샘플 처리 + Peak Detection
# =========================

def process_sample(sample, writer):
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
        noise_levels.popleft()
        scores.popleft()
        boosted_scores.popleft()
        dynamic_thresholds.popleft()

    while peak_times and t - peak_times[0] > PLOT_SEC:
        peak_times.popleft()
        peak_values.popleft()

    # 최근 WINDOW_SEC 구간 추출
    recent_acc = []
    recent_envelope = []
    recent_boosted_score = []

    for tt, aa, ee, bb in zip(times, acc_mags, envelopes, boosted_scores):
        if t - tt <= WINDOW_SEC:
            recent_acc.append(aa)
            recent_envelope.append(ee)
            recent_boosted_score.append(bb)

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

    # 최근 평상시 노이즈 수준 계산
    noise_level = robust_noise(recent_envelope)

    # 평소 노이즈 대비 몇 배 튀었는지 계산
    score = envelope / noise_level

    # score가 1보다 큰 구간만 증폭
    if score > 1:
        boosted_score = 1 + (score - 1) * SCORE_BOOST_GAIN
    else:
        boosted_score = score

    # 핵심: 고정 threshold가 아니라 최근 boosted_score 기반 동적 threshold
    threshold = dynamic_score_threshold(recent_boosted_score)

    times.append(t)
    acc_mags.append(acc_mag)
    vibs.append(vib)
    envelopes.append(envelope)
    noise_levels.append(noise_level)
    scores.append(score)
    boosted_scores.append(boosted_score)
    dynamic_thresholds.append(threshold)

    # peak 판정
    is_above = boosted_score > threshold and envelope > MIN_SIGNAL_G
    is_peak = 0

    if is_above:
        if not in_peak:
            if t - last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                in_peak = True
                peak_candidate_time = t
                peak_candidate_value = boosted_score
                peak_candidate_threshold = threshold
        else:
            if boosted_score > peak_candidate_value:
                peak_candidate_time = t
                peak_candidate_value = boosted_score
                peak_candidate_threshold = threshold

    else:
        if in_peak:
            peak_times.append(peak_candidate_time)
            peak_values.append(peak_candidate_value)

            last_peak_time = peak_candidate_time
            is_peak = 1

            print(
                f"[PEAK] time={datetime.now().strftime('%H:%M:%S.%f')[:-3]} | "
                f"boosted_score={peak_candidate_value:.2f} | "
                f"dynamic_threshold={peak_candidate_threshold:.2f} | "
                f"noise_level={noise_level:.5f} g | "
                f"envelope={envelope:.5f} g"
            )

            in_peak = False
            peak_candidate_time = None
            peak_candidate_value = None
            peak_candidate_threshold = None

    writer.writerow({
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
        "noise_level": noise_level,
        "score": score,
        "boosted_score": boosted_score,
        "dynamic_threshold": threshold,
        "is_above": int(is_above),
        "is_peak": is_peak,
    })


# =========================
# 실시간 그래프
# =========================

fig, ax = plt.subplots(figsize=(12, 5))

line_boosted, = ax.plot([], [], label="boosted score")
line_raw_score, = ax.plot([], [], alpha=0.5, label="raw score")
line_dynamic_th, = ax.plot([], [], linestyle="--", label="dynamic threshold")
peak_scatter = ax.scatter([], [], marker="o", label="peak")

ax.set_title("WT901BLECL Realtime Dynamic Threshold Peak Detection")
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Boosted vibration score")
ax.set_ylim(0, Y_MAX_SCORE)
ax.grid(True)
ax.legend(loc="upper right")


csv_file = None
csv_writer = None


def update_plot(frame):
    while not data_queue.empty():
        sample = data_queue.get()
        process_sample(sample, csv_writer)

    if csv_file is not None:
        csv_file.flush()

    if not times:
        return line_boosted, line_raw_score, line_dynamic_th, peak_scatter

    t0 = times[-1]

    xs = [tt - t0 for tt in times]

    boosted_y = list(boosted_scores)
    raw_score_y = list(scores)
    threshold_y = list(dynamic_thresholds)

    peak_x = [pt - t0 for pt in peak_times]
    peak_y = list(peak_values)

    line_boosted.set_data(xs, boosted_y)
    line_raw_score.set_data(xs, raw_score_y)
    line_dynamic_th.set_data(xs, threshold_y)

    if len(peak_x) > 0:
        peak_scatter.set_offsets(np.column_stack([peak_x, peak_y]))
    else:
        peak_scatter.set_offsets(np.empty((0, 2)))

    ax.set_xlim(-PLOT_SEC, 0)

    return line_boosted, line_raw_score, line_dynamic_th, peak_scatter


def on_close(event):
    stop_event.set()


fig.canvas.mpl_connect("close_event", on_close)


# =========================
# 실행
# =========================

if __name__ == "__main__":
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")

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
        "baseline_acc",
        "vib",
        "envelope",
        "noise_level",
        "score",
        "boosted_score",
        "dynamic_threshold",
        "is_above",
        "is_peak",
    ]

    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    ble_thread = threading.Thread(target=run_ble_thread, daemon=True)
    ble_thread.start()

    ani = FuncAnimation(fig, update_plot, interval=50, blit=False)

    try:
        plt.show()
    finally:
        stop_event.set()
        csv_file.close()
        print("종료됨")
        