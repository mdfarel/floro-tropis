"""Frozen Prithvi-EO-2.0-300M features in the shared feature-file layout (driver shared with extract_terramind.py).

Model    ibm-nasa-geospatial/Prithvi-EO-2.0-300M (Apache-2.0; the non-TL variant, no time/location encoding),
         terratorch 1.2.14 `prithvi_eo_v2_300` = PrithviViT: 24 blocks, dim 1024, patch (1, 16, 16), CLS token.
         Encoder only. Weights: weights/prithvi2-300m/Prithvi_EO_V2_300M.pt (HF revision pinned below; downloaded
         on first use if missing). Runs in .venv-tm.
Input    optical only, 6 HLS bands in Prithvi order Blue, Green, Red, Narrow NIR, SWIR1, SWIR2 = our S2 L2A
         B02 B03 B04 B8A B11 B12 (optical.npy channels 0 1 2 5 6 7; B8A is the narrow NIR that HLS S30 uses).
         Values: reflectance x 10000 (HLS scale), standardised with Prithvi-EO-2.0 mean/std (terratorch
         PRITHVI_V2_MEAN/STD = config.json of the HF repo). PrithviViT does no normalisation itself.
         One time step (num_frames=1; pretraining used 4), 256 px -> 16 x 16 tokens; the fixed 3D sin-cos
         positional embedding is built for (1, 16, 16) (img_size=256), so nothing is interpolated.
         SAR is not used: Prithvi has no SAR input.
Missing  Not availability-aware. A pixel with any NaN band gets 0 in all 6 standardised bands (= band mean; the
         official inference.py likewise sets no-data to ~0 after standardisation). cloud100 is therefore a constant
         input for every patch: the features are encoded once and copied to all rows (sidecar
         constant_input_reused = true); a head can only predict one class map there, by construction.
Output   Layers = zero-based block indices 5, 11, 17, 23 (PANGAEA's choice for Prithvi-EO-2.0-300M); 5/11/17 raw
         block outputs, 23 after the final LayerNorm (terratorch forward_features). CLS token dropped, the 256
         patch tokens reshaped row-major to (1024, 16, 16).

Feature-file contract, clouds, augmentations, CLI: see extract_terramind.py (identical).
Usage: python -I src/extract_prithvi.py --split test --cond cloud50 [--batch 4] [--threads 16] [--limit K]
       [--out-dir data/features] [--patches-dir data/patches] [--weights FILE] [--overwrite]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_terramind import ROOT, PATCH, Encoder, load_checkpoint, run, standardise_fill, tokens_to_grid  # noqa
from sites import S2_BANDS  # noqa: E402

PR_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
PR_REVISION = "9eb1b1102806593963daa333bcc491b1c6f8562f"
PR_FILE = "Prithvi_EO_V2_300M.pt"
PRITHVI_FROM_S2 = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # Blue Green Red NarrowNIR SWIR1 SWIR2


class PrithviEncoder(Encoder):
    name = "prithvi2-300m"
    layers = [5, 11, 17, 23]
    uses_sar = False
    weights_info = {"repo": PR_REPO, "revision": PR_REVISION, "file": PR_FILE, "license": "Apache-2.0",
                    "terratorch_backbone": "prithvi_eo_v2_300", "num_frames": 1}

    def build(self, args):
        import torch
        from terratorch.models.backbones.prithvi_vit import (
            PRETRAINED_BANDS, PRITHVI_V2_MEAN, PRITHVI_V2_STD, _create_prithvi, checkpoint_filter_fn_vit)

        self.mean, self.std = PRITHVI_V2_MEAN, PRITHVI_V2_STD
        self.band_idx = [S2_BANDS.index(b) for b in PRITHVI_FROM_S2]
        ckpt = load_checkpoint(args.weights or ROOT / "weights/prithvi2-300m" / PR_FILE, PR_REPO, PR_FILE,
                               PR_REVISION)
        model = _create_prithvi("prithvi_eo_v2_300", pretrained=False, num_frames=1, img_size=PATCH,
                                encoder_only=True)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
        n_ckpt = len(sd)
        # terratorch's own filter: drops the decoder, keeps the model's fixed sin-cos pos_embed for (1, 16, 16)
        sd = checkpoint_filter_fn_vit(sd, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
        model.load_state_dict(sd, strict=True)
        model.eval()
        self.model = model
        n_params = sum(p.numel() for p in model.parameters())
        self.weights_info = dict(self.weights_info, encoder_params=n_params, tensors_loaded=len(sd))
        print(f"[{self.name}] loaded {len(sd)} encoder tensors of {n_ckpt} in the checkpoint "
              f"({n_params / 1e6:.0f} M params), strict", flush=True)
        return model

    def optical(self, rows, opt8, args):
        return opt8[:, self.band_idx]  # reflectance 0..1, NaN invalid

    def encode(self, optical, sar):
        import torch

        x, _ = standardise_fill(optical * 10000.0, self.mean, self.std)
        feats = self.model.forward_features(torch.from_numpy(x))  # 24 x (B, 1 + 256, 1024), 4D input -> T = 1
        return torch.stack([tokens_to_grid(feats[i], n_prefix=1) for i in self.layers], dim=1)

    def describe(self):
        return ("optical only, HLS-order bands Blue Green Red NarrowNIR SWIR1 SWIR2 = S2 L2A B02 B03 B04 B8A B11 "
                "B12, value = reflectance*10000, standardised with Prithvi-EO-2.0 mean/std (config.json); pixels "
                "with any NaN band set to 0 (band mean); no SAR; 1 time step; 256 px, 3D sin-cos pos-emb built for "
                "(1,16,16); CLS token dropped")


if __name__ == "__main__":
    run(PrithviEncoder())
