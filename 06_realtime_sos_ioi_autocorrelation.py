import asyncio
import csv
import math
import time
import threading
import queue
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bleak import BleakClient


# ============================================================
# 실행 방법
# ============================================================
# 이 파일을 VS Code에서 열고 F5를 누르면 바로 실행됩니다.
#
# 필요 패키지:
# pip install numpy matplotlib bleak
#
# CSV 파일은 이 파이썬 파일과 같은 폴더에 자동 저장됩니다.


# =========================
# 기본 설정값
# =========================

DEVICE_ADDRESS = "DD:D6:0F:01:23:A5"

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "imu_dynamic_threshold_sos.csv"
EVENT_CSV_PATH = BASE_DIR / "imu_sos_event_log.csv"

# 기준값 / 노이즈 계산에 사용할 최근 윈도우 크기
WINDOW_SEC = 5.0

# 그래프에 보여줄 시간 범위
PLOT_SEC = 15.0

# 하나의 충격 안에서 너무 가까운 peak를 중복 검출하지 않기 위한 간격
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

# 최근 boosted_score 분포를 보고 threshold 계산
# 작게 하면 예민, 크게 하면 둔감
DYNAMIC_THRESH_K = 5.0

# threshold 하한 / 상한
MIN_SCORE_THRESHOLD = 15.0
MAX_SCORE_THRESHOLD = 60.0

# 그래프 y축 고정
Y_MAX_SCORE = 80


# =========================
# SOS 패턴 설정값
# =========================

# 구조 신호는 독립된 충격 3회
SOS_REQUIRED_HITS = 3

# 타격과 다음 타격 사이 허용 간격
SOS_MIN_IOI_SEC = 0.25
SOS_MAX_IOI_SEC = 0.80

# 첫 타격부터 마지막 타격까지 허용되는 최대 길이
SOS_PATTERN_MAX_SEC = 2.0

# 두 IOI가 평균 간격에서 얼마나 달라도 허용할지
# 0.30 = 약 30%
SOS_MAX_IOI_ERROR_RATIO = 0.30

# 마지막 충격 뒤 이 시간 동안 새 충격이 없으면
# 하나의 충격 묶음이 끝났다고 판단
SEQUENCE_END_SEC = 0.90

# 자기상관 검사 설정
AUTOCORR_BIN_SEC = 0.01
AUTOCORR_LAG_TOLERANCE_SEC = 0.10
MIN_AUTOCORR_SCORE = 0.50

# 같은 SOS를 연속으로 중복 판정하지 않기 위한 시간
SOS_COOLDOWN_SEC = 3.0


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

# 검출된 모든 peak
peak_times = deque()
peak_values = deque()

# 최종 분류된 이벤트 표시용
sos_event_times = deque()
sos_event_values = deque()

single_impact_times = deque()
single_impact_values = deque()

other_event_times = deque()
other_event_values = deque()

# 현재 분석 중인 충격 묶음
impact_sequence_times = []
impact_sequence_values = []

# envelope 상태
prev_envelope = 0.0

# peak detection 상태
in_peak = False
peak_candidate_time = None
peak_candidate_value = None
peak_candidate_threshold = None
last_peak_time = -999.0

# SOS 중복 방지
last_sos_time = -999.0

# 화면 상태 문구
current_status = "대기 중"

# CSV
csv_file = None
csv_writer = None

event_csv_file = None
event_csv_writer = None


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
            stop_event.set()
            return

        print("\n사용할 notify characteristic:", notify_char)

        def callback(sender, data):
            handle_ble_data(data)

        await client.start_notify(notify_char, callback)

        print("\n데이터 수신 시작")
        print("바닥을 일정한 간격으로 3번 두드리면 SOS 후보로 판단해.")
        print("그래프 창을 닫으면 종료돼.\n")

        while not stop_event.is_set():
            await asyncio.sleep(0.05)

        await client.stop_notify(notify_char)


def run_ble_thread():
    try:
        asyncio.run(ble_main())
    except Exception as exc:
        print(f"\n[BLE 오류] {exc}")
        stop_event.set()


# =========================
# Noise 계산
# =========================

def robust_noise(values):
    """
    최근 window의 envelope 값으로 평소 노이즈 수준 계산.
    median + MAD 방식.
    """

    if len(values) < 10:
        return NOISE_EPS

    arr = np.asarray(values, dtype=float)

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad
    noise_level = med + robust_std

    return max(float(noise_level), NOISE_EPS)


# =========================
# 동적 threshold 계산
# =========================

def dynamic_score_threshold(values):
    """
    최근 boosted_score 분포를 보고 threshold 자동 계산.
    큰 peak가 threshold 계산을 망치지 않도록 상위 10% 제외.
    """

    if len(values) < 10:
        return MIN_SCORE_THRESHOLD

    arr = np.asarray(values, dtype=float)

    upper = np.percentile(arr, 90)
    arr = arr[arr <= upper]

    if len(arr) < 10:
        return MIN_SCORE_THRESHOLD

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    robust_std = 1.4826 * mad

    threshold = med + DYNAMIC_THRESH_K * robust_std
    threshold = max(float(threshold), MIN_SCORE_THRESHOLD)
    threshold = min(threshold, MAX_SCORE_THRESHOLD)

    return threshold


# =========================
# 자기상관
# =========================

def event_autocorrelation_score(hit_times, expected_ioi):
    """
    충격 발생 시각을 0과 1로 된 impulse train으로 변환한 뒤
    예상 IOI 주변의 자기상관 값을 계산한다.

    3회가 거의 같은 간격이면 약 0.67에 가까운 값이 나온다.
    완전히 불규칙하면 값이 낮아진다.
    """

    if len(hit_times) < 3 or expected_ioi <= 0:
        return 0.0

    relative_times = np.asarray(hit_times, dtype=float) - hit_times[0]

    signal_length = int(
        math.ceil((relative_times[-1] + AUTOCORR_BIN_SEC) / AUTOCORR_BIN_SEC)
    ) + 1

    impulse = np.zeros(max(signal_length, 3), dtype=float)

    indices = np.rint(relative_times / AUTOCORR_BIN_SEC).astype(int)
    indices = np.clip(indices, 0, len(impulse) - 1)
    impulse[indices] = 1.0

    autocorr = np.correlate(impulse, impulse, mode="full")
    autocorr = autocorr[len(impulse) - 1:]

    zero_lag = autocorr[0]

    if zero_lag <= 0:
        return 0.0

    expected_lag = int(round(expected_ioi / AUTOCORR_BIN_SEC))
    lag_tolerance = max(
        1,
        int(round(AUTOCORR_LAG_TOLERANCE_SEC / AUTOCORR_BIN_SEC)),
    )

    low = max(1, expected_lag - lag_tolerance)
    high = min(len(autocorr), expected_lag + lag_tolerance + 1)

    if low >= high:
        return 0.0

    # 두 IOI가 약간 달라도 둘 다 예상 lag 주변에 들어오도록
    # 한 점의 최댓값이 아니라 주변 상관값의 합을 사용
    score = np.sum(autocorr[low:high]) / zero_lag

    return float(np.clip(score, 0.0, 1.0))


# =========================
# 이벤트 CSV 기록
# =========================

def write_event_log(
    event_type,
    event_time,
    hit_times,
    hit_values,
    iois=None,
    ioi_error_ratio=None,
    autocorr_score=None,
):
    if event_csv_writer is None:
        return

    if iois is None:
        iois = []

    event_csv_writer.writerow({
        "event_time": event_time,
        "ts_iso": datetime.fromtimestamp(event_time).isoformat(
            timespec="milliseconds"
        ),
        "event_type": event_type,
        "hit_count": len(hit_times),
        "hit_times": "|".join(f"{value:.6f}" for value in hit_times),
        "hit_values": "|".join(f"{value:.3f}" for value in hit_values),
        "iois": "|".join(f"{value:.3f}" for value in iois),
        "ioi_error_ratio": (
            "" if ioi_error_ratio is None else f"{ioi_error_ratio:.4f}"
        ),
        "autocorr_score": (
            "" if autocorr_score is None else f"{autocorr_score:.4f}"
        ),
    })

    if event_csv_file is not None:
        event_csv_file.flush()


# =========================
# 충격 묶음 분석
# =========================

def classify_current_impact_sequence():
    """
    마지막 충격 이후 SEQUENCE_END_SEC 동안 새 충격이 없으면 호출.

    분류:
    - 1회: 단일 충격 후보
    - 규칙적인 3회: SOS
    - 나머지: 일반 반복 진동
    """

    global impact_sequence_times
    global impact_sequence_values
    global last_sos_time
    global current_status

    if not impact_sequence_times:
        return

    hit_times = impact_sequence_times.copy()
    hit_values = impact_sequence_values.copy()

    impact_sequence_times.clear()
    impact_sequence_values.clear()

    hit_count = len(hit_times)
    event_time = hit_times[-1]

    # --------------------------------------------------------
    # 1회 충격: 단순 낙상과 같은 단일 충격 후보
    # IMU만으로 낙상을 확정하지는 않음
    # --------------------------------------------------------
    if hit_count == 1:
        single_impact_times.append(hit_times[0])
        single_impact_values.append(hit_values[0])

        current_status = "단일 충격 후보"

        print(
            f"[단일 충격 후보] "
            f"time={datetime.fromtimestamp(hit_times[0]).strftime('%H:%M:%S.%f')[:-3]} | "
            f"score={hit_values[0]:.2f}"
        )

        write_event_log(
            event_type="SINGLE_IMPACT_CANDIDATE",
            event_time=event_time,
            hit_times=hit_times,
            hit_values=hit_values,
        )
        return

    # --------------------------------------------------------
    # 3회 충격: IOI + 자기상관으로 SOS 검사
    # --------------------------------------------------------
    if hit_count == SOS_REQUIRED_HITS:
        iois = np.diff(np.asarray(hit_times, dtype=float))
        mean_ioi = float(np.mean(iois))
        duration = float(hit_times[-1] - hit_times[0])

        if mean_ioi > 0:
            ioi_error_ratio = float(
                np.max(np.abs(iois - mean_ioi)) / mean_ioi
            )
        else:
            ioi_error_ratio = math.inf

        ioi_range_ok = bool(
            np.all(iois >= SOS_MIN_IOI_SEC)
            and np.all(iois <= SOS_MAX_IOI_SEC)
        )

        duration_ok = duration <= SOS_PATTERN_MAX_SEC

        ioi_regular_ok = (
            ioi_error_ratio <= SOS_MAX_IOI_ERROR_RATIO
        )

        autocorr_score = event_autocorrelation_score(
            hit_times,
            expected_ioi=mean_ioi,
        )

        autocorr_ok = autocorr_score >= MIN_AUTOCORR_SCORE

        cooldown_ok = (
            event_time - last_sos_time >= SOS_COOLDOWN_SEC
        )

        is_sos = (
            ioi_range_ok
            and duration_ok
            and ioi_regular_ok
            and autocorr_ok
            and cooldown_ok
        )

        if is_sos:
            last_sos_time = event_time

            sos_event_times.append(event_time)
            sos_event_values.append(max(hit_values))

            current_status = "🚨 SOS 구조 신호 감지"

            print("\n" + "=" * 58)
            print("🚨 [SOS 구조 신호 감지]")
            print(
                "타격 시각:",
                " / ".join(
                    datetime.fromtimestamp(value).strftime("%H:%M:%S.%f")[:-3]
                    for value in hit_times
                ),
            )
            print(
                f"IOI: {iois[0]:.3f}s / {iois[1]:.3f}s | "
                f"오차율={ioi_error_ratio:.3f}"
            )
            print(f"자기상관 점수: {autocorr_score:.3f}")
            print("=" * 58 + "\n")

            write_event_log(
                event_type="SOS_CONFIRMED",
                event_time=event_time,
                hit_times=hit_times,
                hit_values=hit_values,
                iois=iois,
                ioi_error_ratio=ioi_error_ratio,
                autocorr_score=autocorr_score,
            )
            return

        # 3회였지만 규칙을 통과하지 못함
        other_event_times.append(event_time)
        other_event_values.append(max(hit_values))

        current_status = "불규칙한 3회 충격"

        failed = []

        if not ioi_range_ok:
            failed.append("IOI 범위")
        if not duration_ok:
            failed.append("전체 시간")
        if not ioi_regular_ok:
            failed.append("IOI 규칙성")
        if not autocorr_ok:
            failed.append("자기상관")
        if not cooldown_ok:
            failed.append("SOS 쿨다운")

        print(
            f"[SOS 아님] 3회 충격 | "
            f"IOI={iois[0]:.3f}/{iois[1]:.3f}s | "
            f"오차율={ioi_error_ratio:.3f} | "
            f"자기상관={autocorr_score:.3f} | "
            f"실패={', '.join(failed)}"
        )

        write_event_log(
            event_type="IRREGULAR_THREE_HITS",
            event_time=event_time,
            hit_times=hit_times,
            hit_values=hit_values,
            iois=iois,
            ioi_error_ratio=ioi_error_ratio,
            autocorr_score=autocorr_score,
        )
        return

    # --------------------------------------------------------
    # 2회 또는 4회 이상: 일반 반복 진동
    # 보행은 보통 여러 peak가 계속 이어져 이쪽으로 분류됨
    # --------------------------------------------------------
    other_event_times.append(event_time)
    other_event_values.append(max(hit_values))

    current_status = f"일반 반복 진동 ({hit_count}회)"

    print(
        f"[일반 반복 진동] hit_count={hit_count} | "
        f"duration={hit_times[-1] - hit_times[0]:.3f}s"
    )

    write_event_log(
        event_type="OTHER_REPEATED_VIBRATION",
        event_time=event_time,
        hit_times=hit_times,
        hit_values=hit_values,
    )


def register_impact_peak(peak_time, peak_value):
    """
    검출된 독립 peak를 현재 충격 묶음에 저장.
    이전 peak와 너무 멀면 이전 묶음을 먼저 종료한다.
    """

    if (
        impact_sequence_times
        and peak_time - impact_sequence_times[-1] > SEQUENCE_END_SEC
    ):
        classify_current_impact_sequence()

    impact_sequence_times.append(float(peak_time))
    impact_sequence_values.append(float(peak_value))


def finalize_sequence_if_timeout(current_time):
    """
    마지막 peak 뒤 충분한 정적 구간이 생기면
    현재 충격 묶음을 최종 분류한다.
    """

    if not impact_sequence_times:
        return

    if current_time - impact_sequence_times[-1] >= SEQUENCE_END_SEC:
        classify_current_impact_sequence()


# =========================
# 오래된 그래프 이벤트 제거
# =========================

def remove_old_plot_events(current_time):
    while peak_times and current_time - peak_times[0] > PLOT_SEC:
        peak_times.popleft()
        peak_values.popleft()

    while (
        sos_event_times
        and current_time - sos_event_times[0] > PLOT_SEC
    ):
        sos_event_times.popleft()
        sos_event_values.popleft()

    while (
        single_impact_times
        and current_time - single_impact_times[0] > PLOT_SEC
    ):
        single_impact_times.popleft()
        single_impact_values.popleft()

    while (
        other_event_times
        and current_time - other_event_times[0] > PLOT_SEC
    ):
        other_event_times.popleft()
        other_event_values.popleft()


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
    global current_status

    t = sample["time"]
    acc_mag = sample["acc_mag"]

    # 새로운 peak가 한동안 없으면 직전 충격 묶음을 분류
    finalize_sequence_if_timeout(t)

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

    remove_old_plot_events(t)

    # 최근 WINDOW_SEC 구간 추출
    recent_acc = []
    recent_envelope = []
    recent_boosted_score = []

    for tt, aa, ee, bb in zip(
        times,
        acc_mags,
        envelopes,
        boosted_scores,
    ):
        if t - tt <= WINDOW_SEC:
            recent_acc.append(aa)
            recent_envelope.append(ee)
            recent_boosted_score.append(bb)

    # baseline 계산
    if len(recent_acc) < 10:
        baseline_acc = acc_mag
    else:
        baseline_acc = float(np.median(recent_acc))

    # baseline 제거한 진동 성분
    vib = abs(acc_mag - baseline_acc)

    # envelope 적용
    envelope = (
        ENVELOPE_ALPHA * vib
        + (1 - ENVELOPE_ALPHA) * prev_envelope
    )
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

    # 최근 boosted_score 기반 동적 threshold
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
    is_above = (
        boosted_score > threshold
        and envelope > MIN_SIGNAL_G
    )
    is_peak = 0

    if is_above:
        if not in_peak:
            if t - last_peak_time >= MIN_PEAK_DISTANCE_SEC:
                in_peak = True
                peak_candidate_time = t
                peak_candidate_value = boosted_score
                peak_candidate_threshold = threshold
                current_status = "충격 후보 검출 중"
        else:
            if boosted_score > peak_candidate_value:
                peak_candidate_time = t
                peak_candidate_value = boosted_score
                peak_candidate_threshold = threshold

    else:
        if in_peak:
            peak_times.append(peak_candidate_time)
            peak_values.append(peak_candidate_value)

            register_impact_peak(
                peak_time=peak_candidate_time,
                peak_value=peak_candidate_value,
            )

            last_peak_time = peak_candidate_time
            is_peak = 1
            current_status = (
                f"충격 {len(impact_sequence_times)}회 입력됨"
            )

            print(
                f"[PEAK] "
                f"time={datetime.fromtimestamp(peak_candidate_time).strftime('%H:%M:%S.%f')[:-3]} | "
                f"boosted_score={peak_candidate_value:.2f} | "
                f"dynamic_threshold={peak_candidate_threshold:.2f} | "
                f"noise_level={noise_level:.5f} g"
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
        "current_sequence_hits": len(impact_sequence_times),
        "status": current_status,
    })


# =========================
# 실시간 그래프
# =========================

fig, ax = plt.subplots(figsize=(13, 6))

line_boosted, = ax.plot(
    [],
    [],
    label="boosted score",
)
line_raw_score, = ax.plot(
    [],
    [],
    alpha=0.5,
    label="raw score",
)
line_dynamic_th, = ax.plot(
    [],
    [],
    linestyle="--",
    label="dynamic threshold",
)

peak_scatter = ax.scatter(
    [],
    [],
    marker="o",
    label="detected peak",
)

single_scatter = ax.scatter(
    [],
    [],
    marker="^",
    s=70,
    label="single impact candidate",
)

other_scatter = ax.scatter(
    [],
    [],
    marker="s",
    s=55,
    label="other vibration",
)

sos_scatter = ax.scatter(
    [],
    [],
    marker="*",
    s=180,
    label="SOS",
)

status_text = ax.text(
    0.01,
    0.97,
    "",
    transform=ax.transAxes,
    va="top",
    fontsize=12,
)

ax.set_title(
    "WT901BLECL Realtime SOS Detection "
    "(Dynamic Threshold + IOI + Autocorrelation)"
)
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Boosted vibration score")
ax.set_ylim(0, Y_MAX_SCORE)
ax.grid(True)
ax.legend(loc="upper right")


def set_scatter_data(scatter, x_values, y_values):
    if len(x_values) > 0:
        scatter.set_offsets(
            np.column_stack([x_values, y_values])
        )
    else:
        scatter.set_offsets(np.empty((0, 2)))


def update_plot(frame):
    while not data_queue.empty():
        sample = data_queue.get()
        process_sample(sample, csv_writer)

    if csv_file is not None:
        csv_file.flush()

    if not times:
        return (
            line_boosted,
            line_raw_score,
            line_dynamic_th,
            peak_scatter,
            single_scatter,
            other_scatter,
            sos_scatter,
            status_text,
        )

    t0 = times[-1]

    xs = [tt - t0 for tt in times]

    boosted_y = list(boosted_scores)
    raw_score_y = list(scores)
    threshold_y = list(dynamic_thresholds)

    peak_x = [pt - t0 for pt in peak_times]
    peak_y = list(peak_values)

    single_x = [pt - t0 for pt in single_impact_times]
    single_y = list(single_impact_values)

    other_x = [pt - t0 for pt in other_event_times]
    other_y = list(other_event_values)

    sos_x = [pt - t0 for pt in sos_event_times]
    sos_y = list(sos_event_values)

    line_boosted.set_data(xs, boosted_y)
    line_raw_score.set_data(xs, raw_score_y)
    line_dynamic_th.set_data(xs, threshold_y)

    set_scatter_data(peak_scatter, peak_x, peak_y)
    set_scatter_data(single_scatter, single_x, single_y)
    set_scatter_data(other_scatter, other_x, other_y)
    set_scatter_data(sos_scatter, sos_x, sos_y)

    status_text.set_text(
        f"상태: {current_status}\n"
        f"현재 충격 묶음: {len(impact_sequence_times)}회"
    )

    ax.set_xlim(-PLOT_SEC, 0)

    return (
        line_boosted,
        line_raw_score,
        line_dynamic_th,
        peak_scatter,
        single_scatter,
        other_scatter,
        sos_scatter,
        status_text,
    )


def on_close(event):
    stop_event.set()


fig.canvas.mpl_connect("close_event", on_close)


# =========================
# 실행
# =========================

if __name__ == "__main__":
    sensor_fieldnames = [
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
        "current_sequence_hits",
        "status",
    ]

    event_fieldnames = [
        "event_time",
        "ts_iso",
        "event_type",
        "hit_count",
        "hit_times",
        "hit_values",
        "iois",
        "ioi_error_ratio",
        "autocorr_score",
    ]

    csv_file = open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=sensor_fieldnames,
    )
    csv_writer.writeheader()

    event_csv_file = open(
        EVENT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )
    event_csv_writer = csv.DictWriter(
        event_csv_file,
        fieldnames=event_fieldnames,
    )
    event_csv_writer.writeheader()

    ble_thread = threading.Thread(
        target=run_ble_thread,
        daemon=True,
    )
    ble_thread.start()

    ani = FuncAnimation(
        fig,
        update_plot,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()

    finally:
        stop_event.set()

        # 창을 닫을 때 남아 있는 마지막 충격 묶음도 기록
        classify_current_impact_sequence()

        if csv_file is not None:
            csv_file.close()

        if event_csv_file is not None:
            event_csv_file.close()

        print("\n종료됨")
        print(f"센서 데이터 CSV: {CSV_PATH}")
        print(f"이벤트 로그 CSV: {EVENT_CSV_PATH}")
