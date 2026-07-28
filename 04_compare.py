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

CSV_PATH = "imu_compare_raw_vs_boosted.csv"

# 기준값 / 노이즈 계산에 사용할 최근 윈도우 크기
WINDOW_SEC = 5.0

# 그래프에 보여줄 시간 범위
PLOT_SEC = 15.0

# peak 사이 최소 간격
MIN_PEAK_DISTANCE_SEC = 0.25


# =========================
# 증폭 전 방식 설정값
# =========================

# 기존 vib threshold 민감도
RAW_THRESH_K = 5.0

# 기존 방식 threshold 최소값
RAW_MIN_THRESHOLD_G = 0.03

# 증폭 전 그래프 y축
Y_MAX_RAW_G = 0.6


# =========================
# 증폭 후 방식 설정값
# =========================

# envelope 부드러움 정도
ENVELOPE_ALPHA = 0.3

# boosted_score 기준 threshold
SCORE_THRESHOLD = 30.0

# score 증폭 강도
SCORE_BOOST_GAIN = 16.0

# noise_level이 너무 작아지는 것 방지
NOISE_EPS = 0.0015

# 너무 작은 실제 진동은 peak로 보지 않기
MIN_SIGNAL_G = 0.002

# 증폭 후 그래프 y축
Y_MAX_SCORE = 80


# =========================
# 전역 변수
# =========================

data_queue = queue.Queue()
stop_event = threading.Event()

packet_buffer = bytearray()

# 공통 데이터
times = deque()
acc_mags = deque()
vibs = deque()

# 증폭 전 plot 데이터
raw_thresholds = deque()
raw_peak_times = deque()
raw_peak_values = deque()

# 증폭 후 plot 데이터
envelopes = deque()
scores = deque()
boosted_scores = deque()
score_thresholds = deque()
boost_peak_times = deque()
boost_peak_values = deque()

# envelope 상태
prev_envelope = 0.0

# 증폭 전 peak 상태
raw_in_peak = False
raw_peak_candidate_time = None
raw_peak_candidate_value = None
raw_last_peak_time = -999.0

# 증폭 후 peak 상태
boost_in_peak = False
boost_peak_candidate_time = None
boost_peak_candidate_value = None
boost_last_peak_time = -999.0


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
# Threshold / Noise 계산
# =========================

def robust_raw_threshold(values):
    """
    증폭 전 방식:
    최근 window의 vib 값으로 threshold 계산
    median + K * MAD
    """

    if len(values) < 10:
        return RAW_MIN_THRESHOLD_G

    arr = np.array(values)

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad
    threshold = med + RAW_THRESH_K * robust_std

    return max(threshold, RAW_MIN_THRESHOLD_G)


def robust_noise(values):
    """
    증폭 후 방식:
    최근 window의 envelope 값으로 평소 노이즈 수준 계산
    median + MAD
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
# 샘플 처리 + 두 방식 동시 Peak Detection
# =========================

def process_sample(sample, writer):
    global prev_envelope

    global raw_in_peak
    global raw_peak_candidate_time
    global raw_peak_candidate_value
    global raw_last_peak_time

    global boost_in_peak
    global boost_peak_candidate_time
    global boost_peak_candidate_value
    global boost_last_peak_time

    t = sample["time"]
    acc_mag = sample["acc_mag"]

    # 오래된 데이터 삭제
    while times and t - times[0] > PLOT_SEC:
        times.popleft()
        acc_mags.popleft()
        vibs.popleft()

        raw_thresholds.popleft()

        envelopes.popleft()
        scores.popleft()
        boosted_scores.popleft()
        score_thresholds.popleft()

    while raw_peak_times and t - raw_peak_times[0] > PLOT_SEC:
        raw_peak_times.popleft()
        raw_peak_values.popleft()

    while boost_peak_times and t - boost_peak_times[0] > PLOT_SEC:
        boost_peak_times.popleft()
        boost_peak_values.popleft()

    # 최근 WINDOW_SEC 구간 추출
    recent_acc = []
    recent_vib = []
    recent_envelope = []

    for tt, aa, vv, ee in zip(times, acc_mags, vibs, envelopes):
        if t - tt <= WINDOW_SEC:
            recent_acc.append(aa)
            recent_vib.append(vv)
            recent_envelope.append(ee)

    # baseline 계산
    if len(recent_acc) < 10:
        baseline_acc = acc_mag
    else:
        baseline_acc = np.median(recent_acc)

    # 공통 vib 계산
    vib = abs(acc_mag - baseline_acc)

    # =========================
    # 1) 증폭 전 방식
    # =========================

    raw_threshold = robust_raw_threshold(recent_vib)

    raw_is_above = vib > raw_threshold
    raw_is_peak = 0

    if raw_is_above:
        if not raw_in_peak:
            if t - raw_last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                raw_in_peak = True
                raw_peak_candidate_time = t
                raw_peak_candidate_value = vib
        else:
            if vib > raw_peak_candidate_value:
                raw_peak_candidate_time = t
                raw_peak_candidate_value = vib

    else:
        if raw_in_peak:
            raw_peak_times.append(raw_peak_candidate_time)
            raw_peak_values.append(raw_peak_candidate_value)

            raw_last_peak_time = raw_peak_candidate_time
            raw_is_peak = 1

            print(
                f"[RAW PEAK] time={datetime.now().strftime('%H:%M:%S.%f')[:-3]} | "
                f"vib={raw_peak_candidate_value:.5f} g | "
                f"threshold={raw_threshold:.5f} g"
            )

            raw_in_peak = False
            raw_peak_candidate_time = None
            raw_peak_candidate_value = None

    # =========================
    # 2) 증폭 후 방식
    # =========================

    envelope = ENVELOPE_ALPHA * vib + (1 - ENVELOPE_ALPHA) * prev_envelope
    prev_envelope = envelope

    noise_level = robust_noise(recent_envelope)

    score = envelope / noise_level

    if score > 1:
        boosted_score = 1 + (score - 1) * SCORE_BOOST_GAIN
    else:
        boosted_score = score

    score_threshold = SCORE_THRESHOLD

    boost_is_above = boosted_score > SCORE_THRESHOLD and envelope > MIN_SIGNAL_G
    boost_is_peak = 0

    if boost_is_above:
        if not boost_in_peak:
            if t - boost_last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                boost_in_peak = True
                boost_peak_candidate_time = t
                boost_peak_candidate_value = boosted_score
        else:
            if boosted_score > boost_peak_candidate_value:
                boost_peak_candidate_time = t
                boost_peak_candidate_value = boosted_score

    else:
        if boost_in_peak:
            boost_peak_times.append(boost_peak_candidate_time)
            boost_peak_values.append(boost_peak_candidate_value)

            boost_last_peak_time = boost_peak_candidate_time
            boost_is_peak = 1

            print(
                f"[BOOST PEAK] time={datetime.now().strftime('%H:%M:%S.%f')[:-3]} | "
                f"boosted_score={boost_peak_candidate_value:.2f} | "
                f"raw_score={score:.2f} | "
                f"noise_level={noise_level:.5f} g | "
                f"envelope={envelope:.5f} g"
            )

            boost_in_peak = False
            boost_peak_candidate_time = None
            boost_peak_candidate_value = None

    # =========================
    # plot / csv 데이터 저장
    # =========================

    times.append(t)
    acc_mags.append(acc_mag)
    vibs.append(vib)

    raw_thresholds.append(raw_threshold)

    envelopes.append(envelope)
    scores.append(score)
    boosted_scores.append(boosted_score)
    score_thresholds.append(score_threshold)

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

        "raw_threshold": raw_threshold,
        "raw_is_above": int(raw_is_above),
        "raw_is_peak": raw_is_peak,

        "envelope": envelope,
        "noise_level": noise_level,
        "score": score,
        "boosted_score": boosted_score,
        "score_threshold": score_threshold,
        "boost_is_above": int(boost_is_above),
        "boost_is_peak": boost_is_peak,
    })


# =========================
# 실시간 그래프
# =========================

fig, (ax_raw, ax_boost) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 증폭 전 그래프
line_raw_vib, = ax_raw.plot([], [], label="before boost: vib")
line_raw_th, = ax_raw.plot([], [], linestyle="--", label="before boost threshold")
raw_peak_scatter = ax_raw.scatter([], [], marker="o", label="before boost peak")

ax_raw.set_title("Before Boost: Raw Vibration Peak Detection")
ax_raw.set_ylabel("Vibration magnitude (g)")
ax_raw.set_ylim(0, Y_MAX_RAW_G)
ax_raw.grid(True)
ax_raw.legend(loc="upper right")

# 증폭 후 그래프
line_boosted, = ax_boost.plot([], [], label="after boost: boosted score")
line_score, = ax_boost.plot([], [], alpha=0.5, label="raw score")
line_score_th, = ax_boost.plot([], [], linestyle="--", label="score threshold")
boost_peak_scatter = ax_boost.scatter([], [], marker="o", label="after boost peak")

ax_boost.set_title("After Boost: Boosted Score Peak Detection")
ax_boost.set_xlabel("Time (sec)")
ax_boost.set_ylabel("Boosted vibration score")
ax_boost.set_ylim(0, Y_MAX_SCORE)
ax_boost.grid(True)
ax_boost.legend(loc="upper right")

plt.tight_layout()


csv_file = None
csv_writer = None


def update_plot(frame):
    while not data_queue.empty():
        sample = data_queue.get()
        process_sample(sample, csv_writer)

    if csv_file is not None:
        csv_file.flush()

    if not times:
        return (
            line_raw_vib,
            line_raw_th,
            raw_peak_scatter,
            line_boosted,
            line_score,
            line_score_th,
            boost_peak_scatter,
        )

    t0 = times[-1]
    xs = [tt - t0 for tt in times]

    # 증폭 전 데이터
    raw_y = list(vibs)
    raw_th_y = list(raw_thresholds)

    raw_px = [pt - t0 for pt in raw_peak_times]
    raw_py = list(raw_peak_values)

    line_raw_vib.set_data(xs, raw_y)
    line_raw_th.set_data(xs, raw_th_y)

    if len(raw_px) > 0:
        raw_peak_scatter.set_offsets(np.column_stack([raw_px, raw_py]))
    else:
        raw_peak_scatter.set_offsets(np.empty((0, 2)))

    # 증폭 후 데이터
    boosted_y = list(boosted_scores)
    score_y = list(scores)
    score_th_y = list(score_thresholds)

    boost_px = [pt - t0 for pt in boost_peak_times]
    boost_py = list(boost_peak_values)

    line_boosted.set_data(xs, boosted_y)
    line_score.set_data(xs, score_y)
    line_score_th.set_data(xs, score_th_y)

    if len(boost_px) > 0:
        boost_peak_scatter.set_offsets(np.column_stack([boost_px, boost_py]))
    else:
        boost_peak_scatter.set_offsets(np.empty((0, 2)))

    ax_raw.set_xlim(-PLOT_SEC, 0)
    ax_boost.set_xlim(-PLOT_SEC, 0)

    return (
        line_raw_vib,
        line_raw_th,
        raw_peak_scatter,
        line_boosted,
        line_score,
        line_score_th,
        boost_peak_scatter,
    )


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

        "raw_threshold",
        "raw_is_above",
        "raw_is_peak",

        "envelope",
        "noise_level",
        "score",
        "boosted_score",
        "score_threshold",
        "boost_is_above",
        "boost_is_peak",
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
        