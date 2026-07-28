"""
IMU 기반 3회 바닥 두드림 SOS 신호 감지기

기능
1. CSV에서 3축 가속도 로드
2. 가속도 크기 계산
3. 이동 중앙값 기반 baseline 제거
4. RMS envelope 계산
5. 이동 MAD 기반 동적 threshold 계산
6. 충격 후보 검출 및 잔진동 peak 병합
7. 3회 충격의 IOI 규칙성 + 자기상관 검사
8. SOS 후보와 단일 강한 충격(낙상 후보)을 구분
9. 결과 CSV와 2x2 그래프 저장

필수 패키지
    pip install numpy pandas scipy matplotlib

사용 예시
    python sos_signal_detector.py imu.csv
    python sos_signal_detector.py imu.csv --fs 100
    python sos_signal_detector.py imu.csv --time-col t_rel --ax-col acc_x --ay-col acc_y --az-col acc_z
    python sos_signal_detector.py --demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class Config:
    # 전처리
    baseline_sec: float = 1.5
    envelope_sec: float = 0.06
    noise_window_sec: float = 3.0
    threshold_k: float = 6.0
    min_threshold_quantile: float = 0.70

    # 충격 검출
    min_peak_distance_sec: float = 0.08
    merge_sec: float = 0.20

    # 3회 SOS 규칙
    required_hits: int = 3
    min_ioi_sec: float = 0.20
    max_ioi_sec: float = 0.90
    pattern_max_sec: float = 2.0
    max_ioi_relative_error: float = 0.30

    # 걷기나 연속 진동 오탐 억제
    quiet_before_sec: float = 0.60
    quiet_after_sec: float = 0.70

    # 자기상관
    autocorr_lag_tolerance_sec: float = 0.08
    min_autocorr_score: float = 0.45

    # 단일 강한 충격, 즉 낙상 후보
    fall_isolation_sec: float = 1.0
    fall_peak_ratio: float = 3.0


TIME_CANDIDATES = [
    "time", "timestamp", "t", "t_rel", "elapsed", "elapsed_time",
    "seconds", "second", "sec", "time_sec"
]
AX_CANDIDATES = [
    "acc_x", "ax", "accel_x", "acceleration_x", "accelerationx",
    "linear_acceleration_x", "x_acc"
]
AY_CANDIDATES = [
    "acc_y", "ay", "accel_y", "acceleration_y", "accelerationy",
    "linear_acceleration_y", "y_acc"
]
AZ_CANDIDATES = [
    "acc_z", "az", "accel_z", "acceleration_z", "accelerationz",
    "linear_acceleration_z", "z_acc"
]


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def find_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    label: str,
) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"{label} 컬럼 '{explicit}'을 찾지 못했습니다.\n"
                f"현재 컬럼: {list(df.columns)}"
            )
        return explicit

    normalized_map = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]

    return None


def odd_window(seconds: float, fs: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(seconds * fs)))
    if size % 2 == 0:
        size += 1
    return size


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def estimate_sampling_rate(time_sec: np.ndarray) -> float:
    dt = np.diff(time_sec)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("시간 컬럼에서 유효한 시간 간격을 계산할 수 없습니다.")
    return float(1.0 / np.median(dt))


def load_imu_csv(
    csv_path: Path,
    fs_override: float | None = None,
    time_col: str | None = None,
    ax_col: str | None = None,
    ay_col: str | None = None,
    az_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, str | None]]:
    df = pd.read_csv(csv_path)

    resolved_time = find_column(df, time_col, TIME_CANDIDATES, "시간")
    resolved_ax = find_column(df, ax_col, AX_CANDIDATES, "X축 가속도")
    resolved_ay = find_column(df, ay_col, AY_CANDIDATES, "Y축 가속도")
    resolved_az = find_column(df, az_col, AZ_CANDIDATES, "Z축 가속도")

    missing = []
    if resolved_ax is None:
        missing.append("X축")
    if resolved_ay is None:
        missing.append("Y축")
    if resolved_az is None:
        missing.append("Z축")
    if missing:
        raise ValueError(
            f"가속도 컬럼을 자동으로 찾지 못했습니다: {', '.join(missing)}\n"
            f"현재 컬럼: {list(df.columns)}\n"
            "예: --ax-col acc_x --ay-col acc_y --az-col acc_z"
        )

    numeric_cols = [resolved_ax, resolved_ay, resolved_az]
    if resolved_time is not None:
        numeric_cols.append(resolved_time)

    clean = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(clean) < 10:
        raise ValueError("유효한 데이터가 너무 적습니다.")

    ax = clean[resolved_ax].to_numpy(dtype=float)
    ay = clean[resolved_ay].to_numpy(dtype=float)
    az = clean[resolved_az].to_numpy(dtype=float)

    if resolved_time is not None:
        time_sec = clean[resolved_time].to_numpy(dtype=float)
        time_sec = time_sec - time_sec[0]
        fs = float(fs_override) if fs_override else estimate_sampling_rate(time_sec)
    else:
        if fs_override is None:
            raise ValueError(
                "시간 컬럼을 찾지 못했습니다. 샘플링 주파수를 --fs로 지정해 주세요."
            )
        fs = float(fs_override)
        time_sec = np.arange(len(clean), dtype=float) / fs

    # 시간 역전이나 중복 샘플 제거
    valid = np.ones(len(time_sec), dtype=bool)
    valid[1:] = np.diff(time_sec) > 0
    time_sec = time_sec[valid]
    ax = ax[valid]
    ay = ay[valid]
    az = az[valid]

    columns = {
        "time": resolved_time,
        "ax": resolved_ax,
        "ay": resolved_ay,
        "az": resolved_az,
    }
    return clean, time_sec, ax, ay, az, fs, columns


def preprocess_signal(
    time_sec: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
    cfg: Config,
) -> dict[str, np.ndarray]:
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    baseline_window = odd_window(cfg.baseline_sec, fs)
    baseline = rolling_median(acc_mag, baseline_window)

    vib_signed = acc_mag - baseline
    vib = np.abs(vib_signed)

    envelope_window = odd_window(cfg.envelope_sec, fs)
    envelope = np.sqrt(
        rolling_mean(vib_signed**2, envelope_window)
    )

    noise_window = odd_window(cfg.noise_window_sec, fs)
    noise_median = rolling_median(envelope, noise_window)
    abs_deviation = np.abs(envelope - noise_median)
    noise_mad = rolling_median(abs_deviation, noise_window)

    robust_sigma = 1.4826 * noise_mad
    threshold = noise_median + cfg.threshold_k * robust_sigma

    # MAD가 거의 0인 깨끗한 구간에서도 threshold가 0으로 무너지지 않게 하한 설정
    positive_env = envelope[np.isfinite(envelope) & (envelope > 0)]
    if len(positive_env) > 0:
        threshold_floor = float(
            np.quantile(positive_env, cfg.min_threshold_quantile)
        )
    else:
        threshold_floor = 0.0
    threshold = np.maximum(threshold, threshold_floor)

    return {
        "time": time_sec,
        "acc_mag": acc_mag,
        "baseline": baseline,
        "vib": vib,
        "envelope": envelope,
        "noise_median": noise_median,
        "noise_mad": noise_mad,
        "threshold": threshold,
    }


def detect_raw_peaks(
    envelope: np.ndarray,
    threshold: np.ndarray,
    fs: float,
    cfg: Config,
) -> np.ndarray:
    distance = max(1, int(round(cfg.min_peak_distance_sec * fs)))

    # 배열 형태의 동적 threshold를 각 지점의 최소 높이로 사용
    peaks, _ = find_peaks(
        envelope,
        height=threshold,
        distance=distance,
    )
    return peaks.astype(int)


def merge_peaks(
    peaks: np.ndarray,
    envelope: np.ndarray,
    fs: float,
    cfg: Config,
) -> np.ndarray:
    if len(peaks) == 0:
        return np.array([], dtype=int)

    merge_samples = max(1, int(round(cfg.merge_sec * fs)))
    merged: list[int] = []

    group = [int(peaks[0])]
    for peak in peaks[1:]:
        peak = int(peak)
        if peak - group[-1] <= merge_samples:
            group.append(peak)
        else:
            best = max(group, key=lambda idx: envelope[idx])
            merged.append(best)
            group = [peak]

    best = max(group, key=lambda idx: envelope[idx])
    merged.append(best)

    return np.array(merged, dtype=int)


def binary_autocorrelation_score(
    peak_indices: np.ndarray,
    fs: float,
    expected_ioi_sec: float,
    tolerance_sec: float,
) -> float:
    """
    세 개의 독립 충격을 impulse train으로 만든 뒤,
    예상 IOI 부근의 정규화 자기상관 최댓값을 반환한다.

    정확히 같은 간격으로 3번이면 이론적으로 약 2/3 부근이 나온다.
    """
    if len(peak_indices) < 3:
        return 0.0

    start = int(peak_indices[0])
    end = int(peak_indices[-1])
    length = max(end - start + 1, 3)

    x = np.zeros(length, dtype=float)
    local_indices = peak_indices - start
    local_indices = local_indices[(local_indices >= 0) & (local_indices < length)]
    x[local_indices] = 1.0

    corr = np.correlate(x, x, mode="full")
    corr = corr[length - 1:]
    if corr[0] <= 0:
        return 0.0
    corr = corr / corr[0]

    lag_center = int(round(expected_ioi_sec * fs))
    lag_tol = max(1, int(round(tolerance_sec * fs)))
    low = max(1, lag_center - lag_tol)
    high = min(len(corr), lag_center + lag_tol + 1)

    if low >= high:
        return 0.0

    # 실제 두드림 간격은 몇 샘플 정도 흔들릴 수 있다.
    # 따라서 예상 lag 주변의 자기상관을 한 점만 보지 않고 합산한다.
    # 규칙적인 3회 충격이면 인접한 두 쌍이 같은 lag 부근에 모여
    # 약 2/3 수준의 점수를 만든다.
    return float(min(1.0, np.sum(corr[low:high])))


def detect_sos_patterns(
    merged_peaks: np.ndarray,
    time_sec: np.ndarray,
    envelope: np.ndarray,
    threshold: np.ndarray,
    fs: float,
    cfg: Config,
) -> list[dict]:
    events: list[dict] = []
    n = len(merged_peaks)
    r = cfg.required_hits

    if n < r:
        return events

    used_until = -1

    for start_idx in range(0, n - r + 1):
        if start_idx <= used_until:
            continue

        selected = merged_peaks[start_idx:start_idx + r]
        selected_times = time_sec[selected]
        iois = np.diff(selected_times)

        duration = float(selected_times[-1] - selected_times[0])
        mean_ioi = float(np.mean(iois))
        relative_error = (
            float(np.max(np.abs(iois - mean_ioi)) / mean_ioi)
            if mean_ioi > 0 else math.inf
        )

        prev_gap = math.inf
        if start_idx > 0:
            prev_gap = float(
                selected_times[0] - time_sec[merged_peaks[start_idx - 1]]
            )

        next_gap = math.inf
        if start_idx + r < n:
            next_gap = float(
                time_sec[merged_peaks[start_idx + r]] - selected_times[-1]
            )

        autocorr_score = binary_autocorrelation_score(
            selected,
            fs,
            expected_ioi_sec=mean_ioi,
            tolerance_sec=cfg.autocorr_lag_tolerance_sec,
        )

        conditions = {
            "duration_ok": duration <= cfg.pattern_max_sec,
            "ioi_range_ok": bool(
                np.all(iois >= cfg.min_ioi_sec)
                and np.all(iois <= cfg.max_ioi_sec)
            ),
            "ioi_regular_ok": relative_error <= cfg.max_ioi_relative_error,
            "quiet_before_ok": prev_gap >= cfg.quiet_before_sec,
            "quiet_after_ok": next_gap >= cfg.quiet_after_sec,
            "autocorr_ok": autocorr_score >= cfg.min_autocorr_score,
        }

        is_sos = all(conditions.values())

        if is_sos:
            ratios = envelope[selected] / np.maximum(threshold[selected], 1e-12)
            events.append({
                "label": "SOS",
                "start_time": float(selected_times[0]),
                "end_time": float(selected_times[-1]),
                "peak_indices": selected.copy(),
                "peak_times": selected_times.copy(),
                "iois": iois.copy(),
                "mean_ioi": mean_ioi,
                "ioi_relative_error": relative_error,
                "autocorr_score": autocorr_score,
                "mean_peak_ratio": float(np.mean(ratios)),
                "conditions": conditions,
            })
            used_until = start_idx + r - 1

    return events


def detect_isolated_impacts(
    merged_peaks: np.ndarray,
    time_sec: np.ndarray,
    envelope: np.ndarray,
    threshold: np.ndarray,
    sos_events: list[dict],
    cfg: Config,
) -> list[dict]:
    sos_peak_set: set[int] = set()
    for event in sos_events:
        sos_peak_set.update(int(idx) for idx in event["peak_indices"])

    impacts: list[dict] = []
    peak_times = time_sec[merged_peaks]

    for i, peak_idx in enumerate(merged_peaks):
        peak_idx = int(peak_idx)
        if peak_idx in sos_peak_set:
            continue

        time_i = float(time_sec[peak_idx])
        prev_gap = math.inf if i == 0 else float(time_i - peak_times[i - 1])
        next_gap = math.inf if i == len(merged_peaks) - 1 else float(peak_times[i + 1] - time_i)

        peak_ratio = float(
            envelope[peak_idx] / max(threshold[peak_idx], 1e-12)
        )

        if (
            prev_gap >= cfg.fall_isolation_sec
            and next_gap >= cfg.fall_isolation_sec
            and peak_ratio >= cfg.fall_peak_ratio
        ):
            impacts.append({
                "label": "SINGLE_STRONG_IMPACT",
                "time": time_i,
                "peak_index": peak_idx,
                "peak_ratio": peak_ratio,
            })

    return impacts


def build_peak_table(
    merged_peaks: np.ndarray,
    time_sec: np.ndarray,
    envelope: np.ndarray,
    threshold: np.ndarray,
    sos_events: list[dict],
    isolated_impacts: list[dict],
) -> pd.DataFrame:
    sos_peak_set: set[int] = set()
    for event in sos_events:
        sos_peak_set.update(int(idx) for idx in event["peak_indices"])

    impact_peak_set = {
        int(event["peak_index"]) for event in isolated_impacts
    }

    rows = []
    for number, idx in enumerate(merged_peaks, start=1):
        idx = int(idx)
        if idx in sos_peak_set:
            label = "SOS_HIT"
        elif idx in impact_peak_set:
            label = "SINGLE_STRONG_IMPACT"
        else:
            label = "OTHER_IMPACT"

        rows.append({
            "event_no": number,
            "time_sec": float(time_sec[idx]),
            "envelope": float(envelope[idx]),
            "threshold": float(threshold[idx]),
            "peak_ratio": float(
                envelope[idx] / max(threshold[idx], 1e-12)
            ),
            "label": label,
        })

    return pd.DataFrame(rows)


def build_sos_table(sos_events: list[dict]) -> pd.DataFrame:
    rows = []
    for number, event in enumerate(sos_events, start=1):
        iois = event["iois"]
        rows.append({
            "sos_no": number,
            "start_time_sec": event["start_time"],
            "end_time_sec": event["end_time"],
            "duration_sec": event["end_time"] - event["start_time"],
            "hit_times_sec": ", ".join(f"{x:.3f}" for x in event["peak_times"]),
            "ioi_sec": ", ".join(f"{x:.3f}" for x in iois),
            "mean_ioi_sec": event["mean_ioi"],
            "ioi_relative_error": event["ioi_relative_error"],
            "autocorr_score": event["autocorr_score"],
            "mean_peak_ratio": event["mean_peak_ratio"],
        })
    return pd.DataFrame(rows)


def plot_results(
    signals: dict[str, np.ndarray],
    raw_peaks: np.ndarray,
    merged_peaks: np.ndarray,
    sos_events: list[dict],
    isolated_impacts: list[dict],
    output_path: Path,
    title: str,
) -> None:
    t = signals["time"]
    acc_mag = signals["acc_mag"]
    baseline = signals["baseline"]
    vib = signals["vib"]
    envelope = signals["envelope"]
    threshold = signals["threshold"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    ax = axes[0, 0]
    ax.plot(t, acc_mag, linewidth=1.0, label="acc magnitude")
    ax.plot(t, baseline, linewidth=1.2, label="baseline")
    ax.set_title("1. Acceleration magnitude and baseline")
    ax.set_ylabel("Acceleration")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(t, vib, linewidth=0.9)
    ax.set_title("2. Baseline-removed vibration")
    ax.set_ylabel("|acc magnitude - baseline|")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(t, envelope, linewidth=1.0, label="envelope")
    ax.plot(t, threshold, linewidth=1.2, label="dynamic threshold")
    if len(raw_peaks) > 0:
        ax.scatter(
            t[raw_peaks],
            envelope[raw_peaks],
            s=18,
            marker="x",
            label="raw peaks",
        )
    if len(merged_peaks) > 0:
        ax.scatter(
            t[merged_peaks],
            envelope[merged_peaks],
            s=30,
            label="merged impacts",
        )
    ax.set_title("3. Peak detection and merging")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(t, envelope, linewidth=0.9, label="envelope")
    ax.plot(t, threshold, linewidth=1.0, label="threshold")

    for event_no, event in enumerate(sos_events, start=1):
        times = event["peak_times"]
        indices = event["peak_indices"]
        ax.scatter(
            times,
            envelope[indices],
            s=70,
            marker="o",
            label="SOS hits" if event_no == 1 else None,
        )
        ax.axvspan(
            event["start_time"],
            event["end_time"],
            alpha=0.15,
        )
        ax.text(
            event["start_time"],
            float(np.max(envelope[indices])) * 1.03,
            f"SOS {event_no}\nAC={event['autocorr_score']:.2f}",
            fontsize=9,
        )

    for event_no, impact in enumerate(isolated_impacts, start=1):
        idx = impact["peak_index"]
        ax.scatter(
            [impact["time"]],
            [envelope[idx]],
            s=80,
            marker="^",
            label="single strong impact" if event_no == 1 else None,
        )

    ax.set_title("4. Final classification")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Envelope")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_demo_csv(output_path: Path, fs: float = 100.0) -> None:
    rng = np.random.default_rng(42)
    duration = 14.0
    t = np.arange(0.0, duration, 1.0 / fs)

    ax = rng.normal(0, 0.006, len(t))
    ay = rng.normal(0, 0.006, len(t))
    az = 1.0 + rng.normal(0, 0.006, len(t))

    def add_impact(center_sec: float, amplitude: float, ring_sec: float = 0.25) -> None:
        start = int(center_sec * fs)
        length = max(5, int(ring_sec * fs))
        indices = np.arange(length)
        wave = amplitude * np.exp(-indices / (0.07 * fs)) * np.sin(
            2 * np.pi * 18 * indices / fs
        )
        end = min(len(az), start + length)
        az[start:end] += wave[:end - start]

    # 단일 강한 충격
    add_impact(2.0, 0.45)

    # 정상적인 세 번 SOS
    add_impact(6.0, 0.22)
    add_impact(6.48, 0.20)
    add_impact(6.98, 0.24)

    # 불규칙한 3회, SOS가 아니어야 함
    add_impact(10.0, 0.20)
    add_impact(10.25, 0.20)
    add_impact(11.20, 0.20)

    pd.DataFrame({
        "time": t,
        "acc_x": ax,
        "acc_y": ay,
        "acc_z": az,
    }).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMU 기반 3회 바닥 두드림 SOS 감지"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="테스트용 demo CSV를 생성하고 분석",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="샘플링 주파수. 시간 컬럼이 없을 때 필수",
    )
    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--ax-col", type=str, default=None)
    parser.add_argument("--ay-col", type=str, default=None)
    parser.add_argument("--az-col", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sos_results"),
    )

    # 자주 조절할 파라미터만 CLI에 노출
    parser.add_argument("--threshold-k", type=float, default=6.0)
    parser.add_argument("--merge-sec", type=float, default=0.20)
    parser.add_argument("--min-ioi", type=float, default=0.20)
    parser.add_argument("--max-ioi", type=float, default=0.90)
    parser.add_argument("--pattern-max-sec", type=float, default=2.0)
    parser.add_argument("--max-ioi-error", type=float, default=0.30)
    parser.add_argument("--min-autocorr", type=float, default=0.45)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        csv_path = args.output_dir / "demo_imu.csv"
        make_demo_csv(csv_path)
        print(f"[DEMO] 테스트 CSV 생성: {csv_path}")
    else:
        if args.csv is None:
            raise SystemExit(
                "입력 CSV를 지정하거나 --demo를 사용해 주세요.\n"
                "예: python sos_signal_detector.py imu.csv"
            )
        csv_path = args.csv

    if not csv_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾지 못했습니다: {csv_path}")

    cfg = Config(
        threshold_k=args.threshold_k,
        merge_sec=args.merge_sec,
        min_ioi_sec=args.min_ioi,
        max_ioi_sec=args.max_ioi,
        pattern_max_sec=args.pattern_max_sec,
        max_ioi_relative_error=args.max_ioi_error,
        min_autocorr_score=args.min_autocorr,
    )

    _, time_sec, ax, ay, az, fs, columns = load_imu_csv(
        csv_path=csv_path,
        fs_override=args.fs,
        time_col=args.time_col,
        ax_col=args.ax_col,
        ay_col=args.ay_col,
        az_col=args.az_col,
    )

    signals = preprocess_signal(
        time_sec=time_sec,
        ax=ax,
        ay=ay,
        az=az,
        fs=fs,
        cfg=cfg,
    )

    raw_peaks = detect_raw_peaks(
        envelope=signals["envelope"],
        threshold=signals["threshold"],
        fs=fs,
        cfg=cfg,
    )

    merged_peaks = merge_peaks(
        peaks=raw_peaks,
        envelope=signals["envelope"],
        fs=fs,
        cfg=cfg,
    )

    sos_events = detect_sos_patterns(
        merged_peaks=merged_peaks,
        time_sec=time_sec,
        envelope=signals["envelope"],
        threshold=signals["threshold"],
        fs=fs,
        cfg=cfg,
    )

    isolated_impacts = detect_isolated_impacts(
        merged_peaks=merged_peaks,
        time_sec=time_sec,
        envelope=signals["envelope"],
        threshold=signals["threshold"],
        sos_events=sos_events,
        cfg=cfg,
    )

    peak_table = build_peak_table(
        merged_peaks=merged_peaks,
        time_sec=time_sec,
        envelope=signals["envelope"],
        threshold=signals["threshold"],
        sos_events=sos_events,
        isolated_impacts=isolated_impacts,
    )
    sos_table = build_sos_table(sos_events)

    stem = csv_path.stem
    peak_csv_path = args.output_dir / f"{stem}_impact_events.csv"
    sos_csv_path = args.output_dir / f"{stem}_sos_events.csv"
    plot_path = args.output_dir / f"{stem}_analysis.png"

    peak_table.to_csv(peak_csv_path, index=False)
    sos_table.to_csv(sos_csv_path, index=False)

    plot_results(
        signals=signals,
        raw_peaks=raw_peaks,
        merged_peaks=merged_peaks,
        sos_events=sos_events,
        isolated_impacts=isolated_impacts,
        output_path=plot_path,
        title=f"SOS signal analysis: {csv_path.name}",
    )

    print("\n========== 분석 결과 ==========")
    print(f"입력 파일           : {csv_path}")
    print(f"샘플링 주파수      : {fs:.2f} Hz")
    print(f"사용한 컬럼         : {columns}")
    print(f"원시 peak 수        : {len(raw_peaks)}")
    print(f"병합 후 충격 수     : {len(merged_peaks)}")
    print(f"SOS 감지 수         : {len(sos_events)}")
    print(f"단일 강한 충격 수   : {len(isolated_impacts)}")

    if len(sos_table) > 0:
        print("\n[SOS 이벤트]")
        print(sos_table.to_string(index=False))
    else:
        print("\n[SOS 이벤트 없음]")

    print("\n저장 파일")
    print(f"- 충격 이벤트 CSV   : {peak_csv_path}")
    print(f"- SOS 이벤트 CSV    : {sos_csv_path}")
    print(f"- 분석 그래프       : {plot_path}")


if __name__ == "__main__":
    main()
