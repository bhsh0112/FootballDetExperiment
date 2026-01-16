import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_bbox(center: Tuple[float, float], size: Tuple[float, float]) -> Tuple[int, int, int, int]:
    cx, cy = center
    w, h = size
    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    return x, y, int(round(w)), int(round(h))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def crop_with_padding(
    image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    x0 = int(clamp(x, 0, w_img - 1))
    y0 = int(clamp(y, 0, h_img - 1))
    x1 = int(clamp(x + w, 1, w_img))
    y1 = int(clamp(y + h, 1, h_img))
    crop = image[y0:y1, x0:x1]
    return crop, (x0, y0, x1 - x0, y1 - y0)


def sample_features(gray: np.ndarray, bbox: Tuple[int, int, int, int], max_points: int) -> np.ndarray:
    x, y, w, h = bbox
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=3,
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    points[:, 0, 0] += x
    points[:, 0, 1] += y
    return points.astype(np.float32)


def compute_hist(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    crop, _ = crop_with_padding(image, bbox)
    if crop.size == 0:
        return np.zeros((180, 1), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    if hist_a is None or hist_b is None:
        return 0.0
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def select_roi_or_point(
    frame: np.ndarray, default_size: int
) -> Tuple[int, int, int, int]:
    cv2.namedWindow("select_roi", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("select_roi", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("select_roi")
    if roi is not None and roi[2] > 0 and roi[3] > 0:
        return (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    clicked = {"pt": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (int(x), int(y))

    cv2.namedWindow("select_point", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select_point", on_mouse)
    while True:
        vis = frame.copy()
        if clicked["pt"] is not None:
            cv2.circle(vis, clicked["pt"], 4, (0, 255, 0), -1)
        cv2.imshow("select_point", vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and clicked["pt"] is not None:
            break
        if key == 27:
            break
    cv2.destroyWindow("select_point")
    if clicked["pt"] is None:
        raise RuntimeError("ROI not selected.")
    x, y = clicked["pt"]
    half = max(2, int(default_size // 2))
    h, w = frame.shape[:2]
    x0 = int(clamp(x - half, 0, w - 1))
    y0 = int(clamp(y - half, 0, h - 1))
    x1 = int(clamp(x + half, 1, w))
    y1 = int(clamp(y + half, 1, h))
    return (x0, y0, x1 - x0, y1 - y0)


def track_video(
    input_path: str,
    output_dir: str,
    max_points: int,
    lk_win: int,
    lk_levels: int,
    lk_iters: int,
    max_move: float,
    similarity_thresh: float,
    reinit_every: int,
    roi_size: int,
    show_window: bool,
) -> List[Tuple[int, Optional[float], Optional[float], Optional[int], Optional[int], float, str]]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Failed to read first frame.")
    ensure_dir(output_dir)
    frames_dir = os.path.join(output_dir, "frames_annotated")
    ensure_dir(frames_dir)
    roi = select_roi_or_point(first, roi_size)
    prev_frame = first
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    target_hist = compute_hist(prev_frame, roi)
    prev_points = sample_features(prev_gray, roi, max_points)
    center = bbox_center(roi)
    size = (float(roi[2]), float(roi[3]))
    velocity = (0.0, 0.0)
    results = []
    frame_idx = 0
    while True:
        if frame_idx == 0:
            method = "init"
            confidence = 1.0
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            method = "lk"
            confidence = 0.0
            new_center = None
            if prev_points.size > 0:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    prev_points,
                    None,
                    winSize=(lk_win, lk_win),
                    maxLevel=lk_levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_iters, 0.03),
                )
                if p1 is not None and st is not None:
                    good_new = p1[st[:, 0] == 1]
                    good_old = prev_points[st[:, 0] == 1]
                    if good_new.size > 0:
                        shifts = good_new - good_old
                        median_shift = np.median(shifts.reshape(-1, 2), axis=0)
                        cand = (center[0] + float(median_shift[0]), center[1] + float(median_shift[1]))
                        if max_move <= 0 or np.hypot(cand[0] - center[0], cand[1] - center[1]) <= max_move:
                            new_center = cand
            if new_center is None:
                method = "predict"
                new_center = (center[0] + velocity[0], center[1] + velocity[1])
            pred_bbox = to_bbox(new_center, size)
            hist_now = compute_hist(frame, pred_bbox)
            similarity = hist_similarity(target_hist, hist_now)
            confidence = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            if similarity < similarity_thresh:
                method = "low_conf"
            if frame_idx % max(1, reinit_every) == 0:
                pred_bbox = to_bbox(new_center, size)
                prev_points = sample_features(gray, pred_bbox, max_points)
            velocity = (new_center[0] - center[0], new_center[1] - center[1])
            center = new_center
            prev_frame = frame
            prev_gray = gray
        bbox = to_bbox(center, size)
        frame_to_draw = prev_frame.copy()
        cv2.rectangle(
            frame_to_draw,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_to_draw,
            f"{method} {confidence:.2f}",
            (bbox[0], max(0, bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        out_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame_to_draw)
        results.append(
            (frame_idx, float(center[0]), float(center[1]), int(size[0]), int(size[1]), float(confidence), method)
        )
        if show_window:
            cv2.imshow("tracking", frame_to_draw)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1
    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    return results


def save_csv(
    rows: List[Tuple[int, Optional[float], Optional[float], Optional[int], Optional[int], float, str]],
    output_path: str,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cx", "cy", "w", "h", "confidence", "method"])
        for row in rows:
            writer.writerow(row)


def try_make_video(frames_dir: str, output_path: str, fps: int) -> bool:
    frame_names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith(".jpg")])
    if not frame_names:
        return False
    first = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for name in frame_names:
        frame = cv2.imread(os.path.join(frames_dir, name))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_points", type=int, default=120)
    parser.add_argument("--lk_win", type=int, default=21)
    parser.add_argument("--lk_levels", type=int, default=3)
    parser.add_argument("--lk_iters", type=int, default=20)
    parser.add_argument("--max_move", type=float, default=60.0)
    parser.add_argument("--similarity_thresh", type=float, default=0.2)
    parser.add_argument("--reinit_every", type=int, default=5)
    parser.add_argument("--roi_size", type=int, default=24)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show_window", action="store_true")
    parser.add_argument("--make_video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = track_video(
        input_path=args.input_video,
        output_dir=args.output_dir,
        max_points=args.max_points,
        lk_win=args.lk_win,
        lk_levels=args.lk_levels,
        lk_iters=args.lk_iters,
        max_move=args.max_move,
        similarity_thresh=args.similarity_thresh,
        reinit_every=args.reinit_every,
        roi_size=args.roi_size,
        show_window=args.show_window,
    )
    ensure_dir(args.output_dir)
    save_csv(results, os.path.join(args.output_dir, "tracks.csv"))
    if args.make_video:
        frames_dir = os.path.join(args.output_dir, "frames_annotated")
        try_make_video(frames_dir, os.path.join(args.output_dir, "tracked.mp4"), args.fps)


if __name__ == "__main__":
    main()

