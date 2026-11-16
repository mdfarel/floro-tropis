"""Writes colab/floro_c512.ipynb: rebuilds the FLORO-Tropis data on a Colab GPU and trains the FLORO head with the
PANGAEA-size decoder (512 channels) next to a 128-channel control. Run: python3 colab/make_notebook.py"""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# FLORO-Tropis on a Colab GPU: 512-channel decoder check

This notebook rebuilds the pilot's data from scratch (Sentinel-2/Sentinel-1 from Microsoft Planetary Computer,
Global Mangrove Watch v3 labels), re-extracts frozen FLORO features on the GPU, and trains the UPerNet head twice,
3 seeds each:

* **c128** - the decoder width used in the published pilot (control: shows GPU vs CPU differences)
* **c512** - the PANGAEA decoder width (the check this notebook exists for)

It also checks that the rebuilt patches match the published `data/patches/meta.csv` and `stats.json`.
Runtime: GPU (T4 or better). Expected time: about 2 to 3 hours. At the end a zip with all results downloads to
your computer."""),
    ("code", """# 1. GPU check
import subprocess, torch
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > GPU'
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"""),
    ("code", """# 2. Code (public repository, FLORO as a submodule)
%cd /content
!rm -rf floro-tropis && git clone -q --recurse-submodules https://github.com/mdfarel/floro-tropis.git
%cd /content/floro-tropis
!git log --oneline -1 && git -C vendor/floro log --oneline -1
!mkdir -p /content/published && cp data/patches/meta.csv data/patches/stats.json /content/published/"""),
    ("code", """# 3. Dependencies (Colab already has torch, numpy, scikit-learn, matplotlib)
!apt-get -qq update > /dev/null && apt-get -qq install -y gdal-bin > /dev/null
!pip -q install rasterio odc-stac planetary-computer pystac-client dask safetensors einops huggingface_hub
!gdalbuildvrt --version"""),
    ("code", """# 4. Global Mangrove Watch v3, 2020 layer (CC BY 4.0, Zenodo 6894273) -> data/raw/gmw/gmw2020.vrt
!mkdir -p data/raw/gmw
!curl -sL -o data/raw/gmw/gmw_v3_2020_gtiff.zip "https://zenodo.org/records/6894273/files/gmw_v3_2020_gtiff.zip?download=1"
!python -I -m zipfile -e data/raw/gmw/gmw_v3_2020_gtiff.zip data/raw/gmw/
!cd data/raw/gmw && gdalbuildvrt -q gmw2020.vrt gmw_v3_2020/*.tif && ls -la gmw2020.vrt"""),
    ("code", """# 5. Sentinel-2 / Sentinel-1 composites for the 8 sites (4 parallel jobs, retried on read errors; ~30-60 min)
import subprocess, os, time
groups = [['mahakam', 'langsa'], ['sembilang', 'kubu_raya'], ['bintuni', 'mimika'], ['segara_anakan', 'kendari']]
env = dict(os.environ, GDAL_HTTP_MAX_RETRY='8', GDAL_HTTP_RETRY_DELAY='3')
os.makedirs('logs', exist_ok=True)
def job(sites):
    cmd = ' ; '.join(f'for i in 1 2 3; do python -I src/download.py {s} && break; sleep 20; done' for s in sites)
    return subprocess.Popen(['bash', '-c', cmd], stdout=open(f'logs/dl_{sites[0]}.log', 'w'), stderr=subprocess.STDOUT, env=env)
t0 = time.time(); procs = [job(g) for g in groups]
for p in procs: p.wait()
print(f'{(time.time() - t0) / 60:.1f} min')
!grep -h "done:" logs/dl_*.log
!ls data/sites/*/label.tif | wc -l"""),
    ("code", """# 6. Patches, then compare with the published patch index and train statistics
!python -I src/patches.py | tail -n 25
import json, pandas as pd, numpy as np
a = pd.read_csv('/content/published/meta.csv'); b = pd.read_csv('data/patches/meta.csv')
same_rows = a[['idx', 'site', 'split', 'row', 'col']].equals(b[['idx', 'site', 'split', 'row', 'col']])
mf = float(np.abs(a.mangrove_frac - b.mangrove_frac).max()); sv = float(np.abs(a.s2_valid_frac - b.s2_valid_frac).max())
sa = json.load(open('/content/published/stats.json')); sb = json.load(open('data/patches/stats.json'))
ds = max(abs(sa[m][k]['mean'] - sb[m][k]['mean']) for m in sa for k in sa[m])
print('same patch rows:', same_rows, '| max diff mangrove_frac', mf, '| s2_valid_frac', sv, '| stats mean', ds)
REPRO = {'same_rows': bool(same_rows), 'max_diff_mangrove_frac': mf, 'max_diff_s2_valid_frac': sv, 'max_diff_stats_mean': ds}"""),
    ("code", """# 7. FLORO weights (MIT), pinned revision
!hf download jorlrodriguezg/floro checkpoints/floro_encoder_202603_ep150.safetensors --revision f79d5ae77e35cf32751f36793ff21cae3aaefa0d --local-dir weights > /dev/null
!sha256sum weights/checkpoints/floro_encoder_202603_ep150.safetensors"""),
    ("code", """# 8. Frozen FLORO features on the GPU (train with 4 flips, val, test clean / cloud50 / cloud100)
import time
t0 = time.time()
for split, cond in [('train', 'clean'), ('val', 'clean'), ('test', 'clean'), ('test', 'cloud50'), ('test', 'cloud100')]:
    !python -I src/extract_floro.py --variant s2s1 --split {split} --cond {cond} --device cuda --batch 16 --threads 2 | tail -n 2
print(f'{(time.time() - t0) / 60:.1f} min')"""),
    ("code", """# 9. Heads: 128-channel control and 512-channel PANGAEA width, 3 seeds each (checkpointed per seed)
!mkdir -p results/colab
!python -I src/train_head.py --model floro-s2s1 --seeds 0 1 2 --channels 128 --device cuda --threads 2 --out results/colab/floro-s2s1.c128.gpu.json 2>&1 | grep -E "test clean|test cloud50|early stop|wrote"
!python -I src/train_head.py --model floro-s2s1 --seeds 0 1 2 --channels 512 --device cuda --threads 2 --out results/colab/floro-s2s1.c512.gpu.json 2>&1 | grep -E "test clean|test cloud50|early stop|wrote\""""),
    ("code", """# 10. Summary and download
import json, shutil, glob
out = {'repro_check': REPRO}
for name in ['c128', 'c512']:
    d = json.load(open(f'results/colab/floro-s2s1.{name}.gpu.json'))
    out[name] = {c: {'mean': round(d['conditions'][c]['pooled']['iou_mangrove']['mean'], 4),
                     'sd': round(d['conditions'][c]['pooled']['iou_mangrove']['sd'], 4),
                     'per_seed': [round(r['pooled']['iou_mangrove'], 4) for r in d['conditions'][c]['per_seed']]}
                 for c in d['conditions']}
    out[name]['head_params'] = d['n_params_head']; out[name]['best_epochs'] = d['best_epoch_per_seed']
json.dump(out, open('results/colab/summary_colab.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
for f in glob.glob('data/features/*.json'): shutil.copy(f, 'results/colab/')
shutil.copy('data/patches/meta.csv', 'results/colab/meta_colab.csv'); shutil.copy('data/patches/stats.json', 'results/colab/stats_colab.json')
shutil.make_archive('/content/floro_colab_results', 'zip', 'results/colab')
from google.colab import files
files.download('/content/floro_colab_results.zip')"""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).with_name("floro_c512.ipynb")
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
