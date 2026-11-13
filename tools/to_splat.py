"""점(npz) → 가우시안 스플랫 PLY (INRIA 형식, 17칸 = 알갱이당 68바이트).

사용: python to_splat.py <입력.npz> <출력.ply> [K=1.4] [N_KEEP=0] [CROP=0]
  CROP > 0 이면 카메라들이 바라본 중심에서 수평 반경 CROP(m) 밖 점을 버린다(배경 제거).
"""
import json
import os
import sys
import time

import numpy as np

SH_C0 = 0.28209479177387814
OPACITY = 0.9
HMIN, HMAX = 0.3, 1.3      # 바닥 기준 남길 높이 범위(m). 스프링 기구는 1m 안팎


def look_center(cams: np.ndarray) -> np.ndarray:
    """카메라 시선(+z축)들에 가장 가까운 점 (최소제곱)."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for c in cams:
        o, d = c[:3, 3], c[:3, 2] / np.linalg.norm(c[:3, 2])
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ o
    return np.linalg.solve(a, b)


def write_ply(xyz: np.ndarray, rgb: np.ndarray, path: str, k: float) -> dict:
    n = len(xyz)
    extent = float((xyz.max(0) - xyz.min(0)).max())
    r = k * extent / np.sqrt(n)
    p = xyz.astype(np.float32).copy()
    p[:, 1] *= -1.0                     # 모델은 y가 아래, three.js는 y가 위
    p[:, 2] *= -1.0
    out = np.zeros((n, 17), np.float32)
    out[:, 0:3] = p
    out[:, 6:9] = (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0
    out[:, 9] = np.log(OPACITY / (1 - OPACITY))
    out[:, 10:13] = np.log(r)
    out[:, 13] = 1.0
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
          "".join(f"property float {x}\n" for x in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(out.tobytes())
    size = os.path.getsize(path)
    return {"splats": n, "radius": round(r, 5), "extent_m": round(extent, 2),
            "ply_bytes": size, "bytes_per_splat": round((size - len(hdr)) / n, 2)}


def main(src: str, dst: str, k: str = "1.4", n_keep: str = "0", crop: str = "0") -> None:
    d = np.load(src)
    xyz, rgb, cams = d["xyz"], d["rgb"], d["cams"]
    info = {"points_in": int(len(xyz)), "K": float(k)}
    if float(crop) > 0:
        c = look_center(cams)
        horiz = np.linalg.norm((xyz - c)[:, [0, 2]], axis=1)   # y는 위아래이므로 x, z만
        keep = horiz < float(crop)
        xyz, rgb = xyz[keep], rgb[keep]
        # 바닥 평면(점 대부분이 모래)을 SVD로 맞추고, 바닥에서 HMAX(m) 위의 공중 점(철망 윗부분 등)을 버린다
        ctr = np.median(xyz, axis=0)
        normal = np.linalg.svd(xyz[::50] - ctr, full_matrices=False)[2][2]
        if normal[1] > 0:                                     # 모델 좌표는 y가 아래 → 법선을 위쪽(-y)으로
            normal = -normal
        h = (xyz - ctr) @ normal
        keep = (h > -HMIN) & (h < HMAX)
        xyz, rgb = xyz[keep], rgb[keep]
        # 바닥이 수평이 되게 돌린다: 바닥 법선 → 모델 좌표의 위(-y). 뷰어에서 돌려 볼 때 땅이 기울지 않는다
        up = np.array([0.0, -1.0, 0.0])
        v, cth = np.cross(normal, up), float(normal @ up)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx / (1 + cth)
        xyz = (xyz - ctr) @ rot.T
        info.update(crop_m=float(crop), crop_center=[round(float(v), 2) for v in c],
                    height_keep_m=[-HMIN, HMAX], points_after_crop=int(len(xyz)))
    if int(n_keep) > 0 and int(n_keep) < len(xyz):
        idx = np.linspace(0, len(xyz) - 1, int(n_keep)).astype(int)
        xyz, rgb = xyz[idx], rgb[idx]
    t = time.time()
    info.update(write_ply(xyz, rgb, dst, float(k)))
    info["write_s"] = round(time.time() - t, 2)
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
