"""점구름을 지정한 카메라(학습에 쓴 i번째 카메라) 시점에서 그려 미리보기 이미지로 저장한다.

사용: python preview.py <입력.npz> <출력.jpg> <카메라 번호> [CROP=0]
서버·브라우저 없이 PC와 코랩 결과를 같은 시점에서 비교할 때 쓴다(점 그리기, 스플랫 아님).
"""
import sys

import cv2
import numpy as np

from to_splat import look_center

W, H, FOV_DEG = 960, 540, 65


def main(src: str, dst: str, cam_i: str, crop: str = "0") -> None:
    d = np.load(src)
    xyz, rgb, cams = d["xyz"], d["rgb"], d["cams"]
    if float(crop) > 0:
        c = look_center(cams)
        keep = np.linalg.norm((xyz - c)[:, [0, 2]], axis=1) < float(crop)
        xyz, rgb = xyz[keep], rgb[keep]
    pose = cams[int(cam_i)]                         # cam2world (OpenCV: +z 앞, +y 아래)
    pc = (xyz - pose[:3, 3]) @ pose[:3, :3]          # world → camera
    front = pc[:, 2] > 0.05
    pc, col = pc[front], rgb[front]
    f = (W / 2) / np.tan(np.radians(FOV_DEG) / 2)
    u = (pc[:, 0] / pc[:, 2] * f + W / 2).astype(int)
    v = (pc[:, 1] / pc[:, 2] * f + H / 2).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[ok], v[ok], pc[ok, 2], col[ok]
    order = np.argsort(-z)                          # 먼 것부터 그려 가까운 것이 덮게
    img = np.full((H, W, 3), 18, np.uint8)
    img[v[order], u[order]] = col[order][:, ::-1]
    cv2.imwrite(dst, img)
    print(f"{dst}: 점 {ok.sum():,}개 그림")


if __name__ == "__main__":
    main(*sys.argv[1:])
