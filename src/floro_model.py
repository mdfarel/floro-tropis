"""FLORO encoder (KAUST, MIT licence) wrapped for this pilot, without installing PANGAEA.

The vendored PANGAEA wrapper vendor/floro/pangaea-bench/pangaea/encoders/floro_encoder.py is executed unchanged:
this module only registers a tiny stand-in for `pangaea.encoders.base.Encoder` (the PANGAEA base class, which is
not installed) and then imports the vendored file from disk. All model math, input packing, validity handling and
token-to-grid fusion are the vendored code, byte for byte.

Usage:
    from floro_model import build_floro
    enc = build_floro("s2s1", stats_path="data/patches/stats.json")       # or "s2only"
    feats = enc.encode(optical, sar)  # optical (B, 8, 256, 256), sar (B, 2, 256, 256) float32, NaN = invalid
    # -> list of 4 float32 tensors (B, 1024, 16, 16) for encoder layers [6, 12, 18, 23]

Input scaling (`input_scale`): the default is FLORO's own pretraining normalisation, chosen on a priori grounds (a
development trial on 25 Segara Anakan test patches is disclosed in README, Limitations):
    "floro_z"      DEFAULT. Exactly what FLORO saw in pretraining: the pretraining loader
                   (vendor/floro/src/floro/data/floro_self_supervised_dataloader.py, LoadersPreparation ->
                   NormalizeWithValidity, with the floro.yaml floro_stats passed by get_dataloaders.py) feeds
                   z = (x - mean_floro) / (std_floro + 1e-6) and sets every channel of an invalid band group
                   (BGR, RE, NIR, NIR2, SWIR; SAR VV+VH) to 0 with that group's validity channel 0. Here we
                   range-check against the physical min/max of stats.json (optical 0..1, sar -60..20), z-score
                   with floro_stats, mark the whole band group NaN when any of its bands is invalid, and run the
                   vendored wrapper in its "floro_norm" mode (finite check only, NaN -> 0, validity 0). The
                   dataset mean/std in stats.json are NOT used, so these features do not depend on the train
                   split statistics.
    "dataset_raw"  planned ablation only: the vendored wrapper's own mapping, per band
                   z = (x - mean_ds) / std_ds (stats.json, train split), clip +-5, x = mean_floro + std_floro * z.
                   That puts the encoder input back in PHYSICAL units (reflectance ~0.2, SAR ~-12 / -20 dB, invalid
                   pixels = 0 in those units), which is not how FLORO was pretrained (none of the vendored yamls
                   sets input_scale, so PANGAEA itself runs "floro_norm" on z-scored input).
    "dataset_z"    what a stock PANGAEA run does (NormalizeMeanStd with the dataset mean/std, then the wrapper's
                   default "floro_norm"): z = (x - mean_ds) / std_ds from stats.json, same invalid handling as
                   floro_z. Ablation only.
    Never switch the scale based on test results. extract_floro.py writes non-default scales to a separate
    feature folder (data/features_<scale>) so an ablation can never overwrite the main features.
"""
import hashlib
import importlib.util
import json
import logging
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sites import FLORO_OPTICAL  # noqa: E402

ROOT = HERE.parent
VENDOR_WRAPPER = ROOT / "vendor/floro/pangaea-bench/pangaea/encoders/floro_encoder.py"
DEFAULT_WEIGHTS = ROOT / "weights/checkpoints/floro_encoder_202603_ep150.safetensors"
SAR_NAMES = ["VV", "VH"]
OUTPUT_LAYERS = [6, 12, 18, 23]
MODEL_NAMES = {"s2s1": "floro-s2s1", "s2only": "floro-s2only"}
INPUT_SCALES = ("floro_z", "dataset_raw", "dataset_z")
DEFAULT_INPUT_SCALE = "floro_z"
INPUT_SCALE_RATIONALE = {
    "floro_z": "default, chosen on a priori grounds: identical to FLORO pretraining (NormalizeWithValidity with "
               "floro_stats, invalid band group -> 0 with validity 0); a development trial on 25 test patches is "
               "disclosed in the README",
    "dataset_raw": "planned ablation: vendored PANGAEA-wrapper mapping to FLORO physical units; does not "
                   "match how FLORO was pretrained",
    "dataset_z": "ablation: stock PANGAEA dataset z-score; does not match how FLORO was pretrained",
}
Z_CLIP = 5.0  # vendored FLORO_Wrapper default, used only by dataset_raw
# Band groups that share one validity channel in FLORO pretraining (NormalizeWithValidity / pack_image)
OPTICAL_GROUPS = [["B2", "B3", "B4"], ["B5"], ["B8"], ["B8A"], ["B11", "B12"]]
SAR_GROUPS = [["VV", "VH"]]

# floro.yaml (vendor/floro/pangaea-bench/configs/encoder/floro.yaml), copied verbatim
FLORO_STATS = {
    "optical": {
        "mean": {"B2": 0.2073523, "B3": 0.2670019, "B4": 0.28107855, "B5": 0.24270512, "B8": 0.2793084,
                 "B8A": 0.14631674, "B11": 0.20326711, "B12": 0.14910233},
        "std": {"B2": 0.18876277, "B3": 0.21284738, "B4": 0.20284666, "B5": 0.15357633, "B8": 0.12527913,
                "B8A": 0.17737003, "B11": 0.11302987, "B12": 0.09519225},
    },
    "mods": {
        "mean": {"E": 2094.924, "VV": -11.424427, "VH": -18.914251},
        "std": {"E": 1678.725, "VV": 3.4653134, "VH": 4.3036957},
    },
}
FLORO_ARCH = dict(input_size=256, d_model=1024, patch_size=16, depth=24, num_heads=16, mlp_ratio=4.0,
                  pos_embed_type="absolute", output_dim=1024)
BAND_MAP = {"B": "B2", "G": "B3", "R": "B4", "RE": "B5", "NIR": "B8", "NIR2": "B8A", "SWIR1": "B11", "SWIR2": "B12"}
BAND_MAP_MOD = {"E": None, "VV": "VV", "VH": "VH"}

log = logging.getLogger("floro")


# ---------------------------------------------------------------------------------------------------------------
# Stand-in for pangaea.encoders.base.Encoder (PANGAEA is not installed). Mirrors the attributes the vendored
# wrapper reads (output_layers, encoder_weights, ...) and parameters_warning(); no model math lives here.
# Written for this repository against PANGAEA's Encoder interface; no PANGAEA (GPL-3.0) code is included.
# ---------------------------------------------------------------------------------------------------------------
class _PangaeaEncoderShim(nn.Module):
    def __init__(self, model_name, input_bands, input_size, embed_dim, output_layers, output_dim,
                 multi_temporal, multi_temporal_output, pyramid_output, encoder_weights=None, download_url=None):
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_layers = output_layers
        self.output_dim = output_dim
        self.multi_temporal = multi_temporal
        self.multi_temporal_output = multi_temporal_output
        self.pyramid_output = pyramid_output
        self.encoder_weights = encoder_weights
        self.download_url = download_url

    def parameters_warning(self, missing, incompatible_shape, logger):
        if missing:
            logger.warning("missing parameters: %s", ", ".join(sorted(missing)))
        if incompatible_shape:
            logger.warning("shape-mismatched parameters: %s", ", ".join(sorted(incompatible_shape)))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


def _load_vendored_module():
    name = "pangaea.encoders.floro_encoder"
    if name in sys.modules:
        return sys.modules[name]
    for pkg in ("pangaea", "pangaea.encoders"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    if "pangaea.encoders.base" not in sys.modules:
        base = types.ModuleType("pangaea.encoders.base")
        base.Encoder = _PangaeaEncoderShim
        sys.modules["pangaea.encoders.base"] = base
    spec = importlib.util.spec_from_file_location(name, VENDOR_WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ inside vendor/
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


# ---------------------------------------------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------------------------------------------
def _band(d, name):
    """stats.json band lookup; tolerates B02-style keys."""
    if name in d:
        return d[name]
    alt = name[0] + name[1:].zfill(2) if name[1:].isdigit() else name
    if alt in d:
        return d[alt]
    raise KeyError(f"band {name} not in stats.json ({list(d)})")


def dataset_cfg_from_stats(stats_path):
    """stats.json (contract layout) -> the PANGAEA dataset_cfg dict the vendored wrapper parses for dataset_raw."""
    with open(stats_path) as f:
        st = json.load(f)
    cfg = {"bands": {"optical": list(FLORO_OPTICAL), "sar": list(SAR_NAMES)}}
    for key, field in (("data_mean", "mean"), ("data_std", "std"), ("data_min", "min"), ("data_max", "max")):
        cfg[key] = {
            "optical": [float(_band(st["optical"], b)[field]) for b in FLORO_OPTICAL],
            "sar": [float(_band(st["sar"], b)[field]) for b in SAR_NAMES],
        }
    return cfg


def normalisation_params(variant, stats_path, input_scale):
    """The normalisation numbers that actually reach the encoder for (variant, input_scale): used to fingerprint
    cached features. floro_z uses only floro_stats and the physical min/max; dataset_raw and dataset_z also use
    the dataset mean/std (and dataset_raw the z clip)."""
    cfg = dataset_cfg_from_stats(stats_path)
    mods = ["optical", "sar"] if variant == "s2s1" else ["optical"]
    fields = ("data_min", "data_max") if input_scale == "floro_z" else ("data_mean", "data_std", "data_min", "data_max")
    out = {"input_scale": input_scale, "bands": {m: cfg["bands"][m] for m in mods}}
    for fld in fields:
        out[fld] = {m: cfg[fld][m] for m in mods}
    if input_scale in ("floro_z", "dataset_raw"):
        out["floro_stats"] = {"optical": FLORO_STATS["optical"]}
        if variant == "s2s1":
            out["floro_stats"]["mods"] = {k: {b: FLORO_STATS["mods"][k][b] for b in SAR_NAMES} for k in ("mean", "std")}
    if input_scale == "dataset_raw":
        out["z_clip"] = Z_CLIP
    return out


def normalisation_sha1(variant, stats_path, input_scale):
    blob = json.dumps(normalisation_params(variant, stats_path, input_scale), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------------------------------------------
def describe_input(variant, input_scale):
    """Text for the feature sidecar's \"input\" field."""
    opt = "Sentinel-2 L2A " + ",".join(FLORO_OPTICAL) + " reflectance 0..1"
    sar = " + Sentinel-1 VV,VH dB" if variant == "s2s1" else " (no SAR; Sentinel-1 not used)"
    if input_scale == "floro_z":
        how = ("FLORO-pretraining z=(x-mean_floro)/(std_floro+1e-6) with floro.yaml floro_stats (as "
               "NormalizeWithValidity in FLORO pretraining) after physical range check (optical 0..1, sar -60..20) "
               "and finite check; a band group (BGR, RE, NIR, NIR2, SWIR, SAR VV+VH) with any invalid band -> NaN -> "
               "FLORO_Wrapper input_scale=floro_norm (0 with validity channel 0); dataset mean/std not used")
    elif input_scale == "dataset_raw":
        how = ("ABLATION, not the pretraining regime: FLORO_Wrapper input_scale=dataset_raw: "
               "z=(x-mean_ds)/std_ds from data/patches/stats.json (train split), clip +-5, "
               "x=mean_floro+std_floro*z (physical units); values outside physical min/max (optical 0..1, "
               "sar -60..20) or non-finite -> 0 with validity channel 0")
    else:
        how = ("ABLATION, not the pretraining regime: PANGAEA-default-style z=(x-mean_ds)/std_ds "
               "(data/patches/stats.json, train split) after physical range check (optical 0..1, sar -60..20), "
               "invalid band group -> NaN -> FLORO_Wrapper input_scale=floro_norm (0 with validity channel 0)")
    return opt + sar + "; " + how + "; availability_validity_mode=original, mask ratios 0, absolute pos-emb"


class FloroEncoder:
    """Frozen FLORO encoder. encode() returns 4 tensors (B, 1024, 16, 16) for layers [6, 12, 18, 23]."""

    def __init__(self, wrapper, variant, input_scale, dataset_cfg, weights_path, load_report):
        self.wrapper = wrapper
        self.variant = variant
        self.model_name = MODEL_NAMES[variant]
        self.use_sar = variant == "s2s1"
        self.input_scale = input_scale
        self.dataset_cfg = dataset_cfg
        self.weights_path = str(weights_path)
        self.load_report = load_report
        self.output_layers = list(OUTPUT_LAYERS)
        self.norm_sha1 = None

    def describe_input(self):
        return describe_input(self.variant, self.input_scale)

    def _prep(self, x, modality):
        """Returns a float32 torch tensor ready for the wrapper. floro_z / dataset_z pre-normalise here (NaN marks
        an invalid band group); dataset_raw passes the physical values through (the vendored wrapper maps them)."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32)) if not torch.is_tensor(x) else x.float()
        if self.input_scale == "dataset_raw":
            return x
        names = list(FLORO_OPTICAL) if modality == "optical" else list(SAR_NAMES)
        cfg = self.dataset_cfg
        lo = torch.tensor(cfg["data_min"][modality]).view(1, -1, 1, 1)
        hi = torch.tensor(cfg["data_max"][modality]).view(1, -1, 1, 1)
        if self.input_scale == "floro_z":
            fl = FLORO_STATS["optical" if modality == "optical" else "mods"]
            mu = torch.tensor([fl["mean"][b] for b in names]).view(1, -1, 1, 1)
            sd = torch.tensor([fl["std"][b] for b in names]).view(1, -1, 1, 1)
        else:  # dataset_z
            mu = torch.tensor(cfg["data_mean"][modality]).view(1, -1, 1, 1)
            sd = torch.tensor(cfg["data_std"][modality]).view(1, -1, 1, 1)
        bad = ~torch.isfinite(x) | (x < lo) | (x > hi)
        # FLORO pretraining zeroes every channel of a band group whose shared validity flag is 0, so one invalid
        # band invalidates its whole group (the wrapper's floro_norm path alone would zero only that band).
        groups = OPTICAL_GROUPS if modality == "optical" else SAR_GROUPS
        for grp in groups:
            ch = [names.index(b) for b in grp]
            gbad = bad[:, ch].any(dim=1, keepdim=True)
            bad[:, ch] = gbad.expand(-1, len(ch), -1, -1)
        z = (x - mu) / (sd + 1e-6)  # NormalizeWithValidity, eps 1e-6
        return torch.where(bad, torch.full_like(z, float("nan")), z)

    @torch.inference_mode()
    def encode(self, optical, sar=None):
        image = {"optical": self._prep(optical, "optical")}
        if self.use_sar:
            if sar is None:
                raise ValueError("variant s2s1 needs sar")
            image["sar"] = self._prep(sar, "sar")
        feats = self.wrapper(image)
        assert len(feats) == len(OUTPUT_LAYERS), len(feats)
        return feats

    @torch.inference_mode()
    def pack(self, optical, sar=None):
        """Debug helper: the wrapper's packed 13-channel optical and 5-channel modality tensors."""
        image = {"optical": self._prep(optical, "optical")}
        if self.use_sar and sar is not None:
            image["sar"] = self._prep(sar, "sar")
        return self.wrapper._pack_optical(image), self.wrapper._pack_modalities(image)


def load_weights(wrapper, weights_path):
    """Same key cleaning and matching as the vendored load_encoder_weights, but from safetensors.
    Adapted from FLORO_Wrapper.load_encoder_weights (vendor/floro/pangaea-bench/pangaea/encoders/floro_encoder.py),
    Copyright (c) 2025 Jorge Luis Rodríguez Galvis, MIT licence (vendor/floro/LICENSE).
    Asserts zero missing and zero shape-mismatched encoder keys."""
    from safetensors.torch import load_file

    pretrained = load_file(str(weights_path), device="cpu")
    cleaned = {}
    for k, v in pretrained.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        cleaned[nk] = v
    current = wrapper.encoder.state_dict()
    matched, missing, mismatched = {}, {}, {}
    for name, param in current.items():
        if name not in cleaned:
            missing[name] = tuple(param.shape)
        elif cleaned[name].shape != param.shape:
            mismatched[name] = (tuple(param.shape), tuple(cleaned[name].shape))
        else:
            matched[name] = cleaned[name]
    unexpected = sorted(set(cleaned) - set(current))
    res = wrapper.encoder.load_state_dict(matched, strict=False)
    report = {"checkpoint_keys": len(cleaned), "encoder_keys": len(current), "loaded": len(matched),
              "missing": len(missing), "shape_mismatch": len(mismatched), "unexpected": len(unexpected)}
    print(f"[floro] weights {Path(weights_path).name}: " + ", ".join(f"{k}={v}" for k, v in report.items()),
          flush=True)
    assert not missing, f"missing encoder keys: {sorted(missing)[:10]}"
    assert not mismatched, f"shape-mismatched encoder keys: {mismatched}"
    assert not res.missing_keys and not res.unexpected_keys, res
    del pretrained, cleaned
    return report


def build_floro(variant, stats_path, weights_path=DEFAULT_WEIGHTS, input_scale=DEFAULT_INPUT_SCALE):
    if variant not in MODEL_NAMES:
        raise ValueError(f"variant must be one of {list(MODEL_NAMES)}")
    if input_scale not in INPUT_SCALES:
        raise ValueError(f"input_scale must be one of {INPUT_SCALES}")
    vend = _load_vendored_module()
    use_sar = variant == "s2s1"
    dataset_cfg = dataset_cfg_from_stats(stats_path)
    input_bands = {"optical": list(FLORO_OPTICAL)}
    if use_sar:
        input_bands["sar"] = list(SAR_NAMES)
    wrapper = vend.FLORO_Wrapper(
        encoder_weights=str(weights_path),
        input_size=FLORO_ARCH["input_size"],
        input_bands=input_bands,
        output_layers=list(OUTPUT_LAYERS),
        output_dim=FLORO_ARCH["output_dim"],
        download_url=None,
        d_model=FLORO_ARCH["d_model"],
        patch_size=FLORO_ARCH["patch_size"],
        depth=FLORO_ARCH["depth"],
        num_heads=FLORO_ARCH["num_heads"],
        mlp_ratio=FLORO_ARCH["mlp_ratio"],
        pos_embed_type=FLORO_ARCH["pos_embed_type"],
        mask_ratio_ms=0.0,
        mask_ratio_mods=0.0,
        band_map=dict(BAND_MAP),
        band_map_mod=dict(BAND_MAP_MOD),
        use_dem=False,
        use_sar=use_sar,
        input_scale="dataset_raw" if input_scale == "dataset_raw" else "floro_norm",
        floro_stats=FLORO_STATS,
        z_clip=Z_CLIP,
        dataset_cfg=dataset_cfg,
        availability_validity_mode="original",
    )
    report = load_weights(wrapper, weights_path)
    wrapper.eval()
    wrapper.freeze()
    enc = FloroEncoder(wrapper, variant, input_scale, dataset_cfg, weights_path, report)
    enc.norm_sha1 = normalisation_sha1(variant, stats_path, input_scale)
    return enc


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Load FLORO and run one random batch (sanity check).")
    ap.add_argument("--variant", choices=list(MODEL_NAMES), default="s2s1")
    ap.add_argument("--stats", default=str(ROOT / "data/patches/stats.json"))
    ap.add_argument("--input-scale", choices=list(INPUT_SCALES), default=DEFAULT_INPUT_SCALE)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    enc = build_floro(a.variant, a.stats, input_scale=a.input_scale)
    rng = np.random.default_rng(0)
    opt = rng.uniform(0, 0.4, (2, 8, 256, 256)).astype(np.float32)
    sar = rng.uniform(-25, -5, (2, 2, 256, 256)).astype(np.float32)
    t = time.time()
    out = enc.encode(opt, sar)
    print([tuple(o.shape) for o in out], f"{time.time() - t:.1f} s")
