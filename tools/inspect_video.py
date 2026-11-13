"""영상 정보 확인 + 프레임 24장 고르게 추출 + 흐림 점수 + 미리보기 격자.

사용: python tools/inspect_video.py raw/영상.mp4 frames/이름
"""
import sys
from pathlib import Path

import cv2
import numpy as np

N_FRAMES = 24
LONG_SIDE = 1024          # 저장용 긴 변 (모델은 다시 줄여 넣는다)
THUMB_W = 240


def main(video: str, out_dir: str) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"해상도 {w}x{h}, fps {fps:.1f}, 프레임 {total}, 길이 {total / fps:.1f}초")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, total - 1, N_FRAMES).round().astype(int)
    thumbs, sharp = [], []
    for i, k in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, frame = cap.read()
        if not ok:
            print(f"  프레임 {k} 읽기 실패")
            continue
        s = LONG_SIDE / max(frame.shape[:2])
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{i:02d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        sharp.append(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        t = THUMB_W / small.shape[1]
        thumb = cv2.resize(small, None, fx=t, fy=t)
        cv2.putText(thumb, f"{i:02d} {sharp[-1]:.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        thumbs.append(thumb)
    cap.release()

    med = float(np.median(sharp))
    blurry = [i for i, v in enumerate(sharp) if v < med * 0.4]
    print(f"선명도 중앙값 {med:.0f}, 흐린 장(중앙값 40% 미만): {blurry or '없음'}")

    cols = 6
    th, tw = thumbs[0].shape[:2]
    grid = np.zeros((th * ((len(thumbs) + cols - 1) // cols), tw * cols, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    cv2.imwrite(str(out / "_grid.jpg"), grid)
    print(f"저장: {out}/00~{len(thumbs) - 1:02d}.jpg, 미리보기 {out}/_grid.jpg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
