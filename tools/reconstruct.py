"""사진 24장 → MapAnything → 점·색·카메라 (PC와 코랩에서 같은 코드로 돌려 비교한다).

사용: python reconstruct.py <프레임 폴더> <출력 폴더> <라벨> [fp16|bf16]
출력: <출력>/<라벨>.npz (xyz, rgb, cams), <출력>/<라벨>.json (측정값)
"""
import glob
import json
import os
import platform
import sys
import time

import numpy as np
import torch

MODEL_ID = "facebook/map-anything-apache"
CLIP_PCT = 0.5            # 각 축 위아래 0.5% 밖 점 제거


def main(frame_dir: str, out_dir: str, label: str, dtype: str = "fp16") -> None:
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    assert torch.cuda.is_available(), "GPU가 없습니다"
    prop = torch.cuda.get_device_properties(0)
    paths = sorted(p for p in glob.glob(os.path.join(frame_dir, "*.jpg")) if not os.path.basename(p).startswith("_"))
    m = {"label": label, "gpu": prop.name, "gpu_mem_gb": round(prop.total_memory / 1e9, 1),
         "torch": torch.__version__, "os": platform.system(), "dtype": dtype, "frames": len(paths)}

    t = time.time()
    model = MapAnything.from_pretrained(MODEL_ID).to("cuda").eval()
    m["model_load_s"] = round(time.time() - t, 1)

    views = load_images(paths)
    m["model_input_hw"] = list(views[0]["img"].shape[-2:])

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        pred = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                           use_amp=True, amp_dtype=dtype,
                           apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    torch.cuda.synchronize()
    m["infer_s"] = round(time.time() - t, 2)
    m["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    pts, cols, cams = [], [], []
    for v in pred:
        mask = v["mask"][0, ..., 0].cpu().numpy()
        pts.append(v["pts3d"][0].float().cpu().numpy()[mask])
        cols.append((v["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[mask])
        cams.append(v["camera_poses"][0].float().cpu().numpy())
    xyz, rgb, cams = np.concatenate(pts), np.concatenate(cols), np.stack(cams)
    m["points_raw"] = int(len(xyz))

    lo, hi = np.percentile(xyz, CLIP_PCT, axis=0), np.percentile(xyz, 100 - CLIP_PCT, axis=0)
    keep = np.all((xyz > lo) & (xyz < hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    m["points"] = int(len(xyz))

    cpos = cams[:, :3, 3]
    cam_diag = float(np.linalg.norm(cpos.max(0) - cpos.min(0)))
    scene_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    m["camera_spread_ratio"] = round(cam_diag / scene_diag, 4)
    m["scene_diag_m"] = round(scene_diag, 2)
    m["cam_path_m"] = round(float(np.linalg.norm(np.diff(cpos, axis=0), axis=1).sum()), 2)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{label}.npz"), xyz=xyz, rgb=rgb, cams=cams)
    with open(os.path.join(out_dir, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
