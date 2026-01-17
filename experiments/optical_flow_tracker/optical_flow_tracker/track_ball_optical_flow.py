import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def list_images(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    names = [
        name
        for name in os.listdir(input_dir)
        if os.path.splitext(name.lower())[1] in exts
    ]
    names.sort()
    return [os.path.join(input_dir, name) for name in names]


def detect_candidates(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    min_area: float,
    max_area: float,
    diff_thresh: int,
    blur_ksize: int,
    morph_ksize: int,
) -> List[Tuple[float, float, float, float]]:
    diff = cv2.absdiff(prev_gray, gray)
    if blur_ksize > 0:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    _, bw = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if morph_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        candidates.append((cx, cy, area, circularity))
    return candidates


def choose_candidate(
    candidates: List[Tuple[float, float, float, float]],
    predicted: Optional[Tuple[float, float]],
    max_candidate_dist: float,
) -> Optional[Tuple[float, float]]:
    if not candidates:
        return None
    if predicted is None:
        candidates.sort(key=lambda x: (-x[3], x[2]))
        return (candidates[0][0], candidates[0][1])
    px, py = predicted
    best = None
    best_dist = None
    for cx, cy, _, _ in candidates:
        dist = float(np.hypot(cx - px, cy - py))
        if best is None or dist < best_dist:
            best = (cx, cy)
            best_dist = dist
    if best is None or best_dist is None:
        return None
    if max_candidate_dist > 0 and best_dist > max_candidate_dist:
        return None
    return best


def track_sequence(
    image_paths: List[str],
    output_dir: str,
    min_area: float,
    max_area: float,
    diff_thresh: int,
    blur_ksize: int,
    morph_ksize: int,
    lk_win: int,
    lk_levels: int,
    lk_iters: int,
    max_move: float,
    max_candidate_dist: float,
) -> List[Tuple[str, Optional[float], Optional[float], str]]:
    if not image_paths:
        return []
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames_annotated")
    os.makedirs(frames_dir, exist_ok=True)
    prev_gray = None
    prev_point = None
    velocity = None
    results = []
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            results.append((os.path.basename(path), None, None, "read_fail"))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        method = "init"
        point = None
        if prev_gray is not None:
            predicted = None
            if prev_point is not None and velocity is not None:
                predicted = (prev_point[0] + velocity[0], prev_point[1] + velocity[1])
            lk_valid = False
            if prev_point is not None:
                p0 = np.array([[prev_point]], dtype=np.float32)
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    p0,
                    None,
                    winSize=(lk_win, lk_win),
                    maxLevel=lk_levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_iters, 0.03),
                )
                if st is not None and st[0][0] == 1:
                    cand = p1[0][0]
                    if max_move <= 0 or np.hypot(cand[0] - prev_point[0], cand[1] - prev_point[1]) <= max_move:
                        point = (float(cand[0]), float(cand[1]))
                        lk_valid = True
                        method = "lk"
            if not lk_valid:
                candidates = detect_candidates(
                    prev_gray,
                    gray,
                    min_area,
                    max_area,
                    diff_thresh,
                    blur_ksize,
                    morph_ksize,
                )
                point = choose_candidate(candidates, predicted, max_candidate_dist)
                method = "cand" if point is not None else "none"
        if prev_gray is None:
            method = "seed"
        if prev_gray is not None and point is not None:
            if prev_point is not None:
                velocity = (point[0] - prev_point[0], point[1] - prev_point[1])
            prev_point = point
        elif prev_gray is None:
            prev_point = None
            velocity = None
        else:
            if prev_point is not None and velocity is not None:
                prev_point = (prev_point[0] + velocity[0], prev_point[1] + velocity[1])
                method = "predict"
                point = prev_point
        if point is None:
            results.append((os.path.basename(path), None, None, method))
        else:
            results.append((os.path.basename(path), point[0], point[1], method))
            cv2.circle(frame, (int(point[0]), int(point[1])), 6, (0, 255, 0), 2)
        out_path = os.path.join(frames_dir, os.path.basename(path))
        cv2.imwrite(out_path, frame)
        prev_gray = gray
    return results


def save_csv(rows: List[Tuple[str, Optional[float], Optional[float], str]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "method"])
        for row in rows:
            writer.writerow(row)


def try_make_gif(frames_dir: str, output_path: str, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        return False
    frame_paths = list_images(frames_dir)
    if not frame_paths:
        return False
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(output_path, images, fps=fps)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_area", type=float, default=6.0)
    parser.add_argument("--max_area", type=float, default=300.0)
    parser.add_argument("--diff_thresh", type=int, default=18)
    parser.add_argument("--blur_ksize", type=int, default=5)
    parser.add_argument("--morph_ksize", type=int, default=3)
    parser.add_argument("--lk_win", type=int, default=21)
    parser.add_argument("--lk_levels", type=int, default=3)
    parser.add_argument("--lk_iters", type=int, default=20)
    parser.add_argument("--max_move", type=float, default=60.0)
    parser.add_argument("--max_candidate_dist", type=float, default=80.0)
    parser.add_argument("--gif_fps", type=int, default=8)
    parser.add_argument("--make_gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.input_dir)
    results = track_sequence(
        image_paths=image_paths,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_area=args.max_area,
        diff_thresh=args.diff_thresh,
        blur_ksize=args.blur_ksize,
        morph_ksize=args.morph_ksize,
        lk_win=args.lk_win,
        lk_levels=args.lk_levels,
        lk_iters=args.lk_iters,
        max_move=args.max_move,
        max_candidate_dist=args.max_candidate_dist,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "tracks.csv")
    save_csv(results, csv_path)
    if args.make_gif:
        frames_dir = os.path.join(args.output_dir, "frames_annotated")
        gif_path = os.path.join(args.output_dir, "tracked.gif")
        try_make_gif(frames_dir, gif_path, args.gif_fps)


if __name__ == "__main__":
    main()

