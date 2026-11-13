"""코랩용 노트북(colab_t4.ipynb)과 업로드 묶음(colab_upload.zip)을 만든다."""
import json
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
frames = root / "frames" / sys.argv[1]

with zipfile.ZipFile(root / "colab_upload.zip", "w") as z:
    for p in sorted(frames.glob("[0-9][0-9].jpg")):
        z.write(p, f"frames/{p.name}")
    z.write(root / "tools" / "reconstruct.py", "reconstruct.py")

cells = [
    ("markdown", "# 7장 과제: 코랩 T4 복원 (PC와 비교용)\n\n**런타임 → 런타임 유형 변경 → T4 GPU** 로 바꾼 뒤 **런타임 → 모두 실행**. 중간에 파일 고르는 창이 뜨면 `colab_upload.zip`을 고릅니다. 끝나면 결과 두 파일이 내려받아집니다."),
    ("code", "import torch, time\nT0 = time.time()\nassert torch.cuda.is_available(), '런타임 유형을 T4 GPU로 바꾸세요'\nprint(torch.cuda.get_device_name(0), torch.__version__)"),
    ("code", "%%capture\n!git clone -q --depth 1 https://github.com/facebookresearch/map-anything.git\n!cd map-anything && pip install -q -e ."),
    ("code", "import sys; sys.path.insert(0, '/content/map-anything')\nimport mapanything\nINSTALL_S = time.time() - T0\nprint(f'설치 {INSTALL_S:.0f}초')"),
    ("code", "from google.colab import files\nup = files.upload()\n!unzip -o -q colab_upload.zip -d /content/work\n!ls /content/work/frames | wc -l"),
    ("code", "import json\n!cd /content/work && python reconstruct.py frames out colab_t4 fp16\nm = json.load(open('/content/work/out/colab_t4.json'))\nm['install_s'] = round(INSTALL_S, 1)\nm['total_s'] = round(time.time() - T0, 1)\njson.dump(m, open('/content/work/out/colab_t4.json', 'w'), ensure_ascii=False, indent=1)\nprint(m)"),
    ("code", "files.download('/content/work/out/colab_t4.json')\nfiles.download('/content/work/out/colab_t4.npz')"),
]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells]}
(root / "colab_t4.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", root / "colab_t4.ipynb", root / "colab_upload.zip")
