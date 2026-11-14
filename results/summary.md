# FLORO-Tropis pilot: test results

4 unseen test sites (Segara Anakan, Kubu Raya, Mimika, Kendari), 400 patches of 256 x 256 px at 10 m, 26,214,400 pixels, mangrove share 0.211 (Global Mangrove Watch v3 2020). Mean ± sd over seeds. Brackets: 95% patch-bootstrap interval of the seed-mean pooled IoU (stratified by site, 2000 draws), only where per-patch counts exist.

## Clean imagery

| Model | Seeds | Mangrove IoU | mIoU | MCC | Precision | Recall | Site-mean IoU | IoU without Segara Anakan |
|---|---|---|---|---|---|---|---|---|
| Per-pixel gradient boosting (S2+S1) | 3 | 0.759 ± 0.003 [0.726–0.790] | 0.842 ± 0.002 | 0.825 ± 0.003 | 0.825 ± 0.003 | 0.904 ± 0.002 | 0.653 ± 0.003 | 0.843 ± 0.001 |
| U-Net from scratch (S2+S1) | 5 | 0.795 ± 0.014 | 0.868 ± 0.009 | 0.856 ± 0.011 | 0.894 ± 0.024 | 0.878 ± 0.019 | 0.675 ± 0.013 | 0.872 ± 0.008 |
| FLORO frozen + head (S2+S1) | 3 | 0.785 ± 0.012 | 0.863 ± 0.007 | 0.849 ± 0.009 | 0.906 ± 0.006 | 0.855 ± 0.009 | 0.664 ± 0.012 | 0.862 ± 0.014 |
| FLORO frozen + head (S2 only) | 3 | 0.769 ± 0.023 [0.732–0.802] | 0.852 ± 0.015 | 0.838 ± 0.018 | 0.915 ± 0.015 | 0.827 ± 0.014 | 0.639 ± 0.017 | 0.844 ± 0.011 |
| TerraMind-L frozen + head (S2 12 bands + S1) | 3 | 0.803 ± 0.013 | 0.874 ± 0.008 | 0.864 ± 0.008 | 0.931 ± 0.010 | 0.854 ± 0.023 | 0.687 ± 0.024 | 0.864 ± 0.010 |
| Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1) | 3 | 0.779 ± 0.012 [0.741–0.814] | 0.859 ± 0.007 | 0.847 ± 0.008 | 0.930 ± 0.002 | 0.827 ± 0.015 | 0.647 ± 0.010 | 0.846 ± 0.008 |
| Ablation: FLORO S2+S1 with dataset_raw input scaling | 3 | 0.701 ± 0.026 [0.659–0.740] | 0.804 ± 0.019 | 0.776 ± 0.024 | 0.814 ± 0.041 | 0.834 ± 0.009 | 0.596 ± 0.027 | 0.791 ± 0.036 |
| Constant map: all mangrove | – | 0.211 | 0.106 | 0 | 0.211 | 1 | – | – |
| Constant map: all non-mangrove | – | 0 | 0.394 | 0 | – | 0 | – | – |

## 50% cloud (simulated, identical masks for all models)

| Model | Seeds | Mangrove IoU | mIoU | MCC | Precision | Recall | Site-mean IoU | IoU without Segara Anakan |
|---|---|---|---|---|---|---|---|---|
| Per-pixel gradient boosting (S2+S1) | 3 | 0.370 ± 0.001 [0.330–0.405] | 0.512 ± 0.007 | 0.405 ± 0.005 | 0.408 ± 0.007 | 0.801 ± 0.031 | 0.326 ± 0.004 | 0.383 ± 0.002 |
| U-Net from scratch (S2+S1) | 5 | 0.501 ± 0.132 | 0.660 ± 0.099 | 0.594 ± 0.123 | 0.702 ± 0.201 | 0.697 ± 0.194 | 0.423 ± 0.109 | 0.539 ± 0.157 |
| FLORO frozen + head (S2+S1) | 3 | 0.639 ± 0.009 | 0.760 ± 0.005 | 0.719 ± 0.008 | 0.741 ± 0.001 | 0.823 ± 0.015 | 0.532 ± 0.020 | 0.734 ± 0.023 |
| FLORO frozen + head (S2 only) | 3 | 0.324 ± 0.124 [0.300–0.347] | 0.582 ± 0.069 | 0.496 ± 0.076 | 0.921 ± 0.097 | 0.344 ± 0.156 | 0.247 ± 0.108 | 0.358 ± 0.140 |
| TerraMind-L frozen + head (S2 12 bands + S1) | 3 | 0.683 ± 0.101 | 0.798 ± 0.060 | 0.774 ± 0.069 | 0.889 ± 0.064 | 0.758 ± 0.153 | 0.599 ± 0.092 | 0.730 ± 0.105 |
| Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1) | 3 | 0.609 ± 0.035 [0.574–0.641] | 0.749 ± 0.020 | 0.704 ± 0.024 | 0.831 ± 0.020 | 0.698 ± 0.060 | 0.504 ± 0.024 | 0.656 ± 0.038 |
| Ablation: FLORO S2+S1 with dataset_raw input scaling | 3 | 0.349 ± 0.103 [0.311–0.385] | 0.443 ± 0.163 | 0.371 ± 0.140 | 0.388 ± 0.156 | 0.857 ± 0.116 | 0.319 ± 0.053 | 0.362 ± 0.119 |
| Constant map: all mangrove | – | 0.211 | 0.106 | 0 | 0.211 | 1 | – | – |
| Constant map: all non-mangrove | – | 0 | 0.394 | 0 | – | 0 | – | – |

## Optical missing (radar only)

| Model | Seeds | Mangrove IoU | mIoU | MCC | Precision | Recall | Site-mean IoU | IoU without Segara Anakan |
|---|---|---|---|---|---|---|---|---|
| Per-pixel gradient boosting (S2+S1) | 3 | 0.222 ± 0.009 [0.193–0.249] | 0.309 ± 0.012 | 0.104 ± 0.018 | 0.246 ± 0.003 | 0.697 ± 0.062 | 0.224 ± 0.012 | 0.221 ± 0.008 |
| Per-pixel gradient boosting, radar only (reference) | 3 | 0.250 ± 0.000 [0.217–0.283] | 0.244 ± 0.000 | 0.205 ± 0.000 | 0.253 ± 0.000 | 0.960 ± 0.001 | 0.258 ± 0.000 | 0.259 ± 0.000 |
| U-Net from scratch (S2+S1) | 5 | 0.148 ± 0.111 | 0.306 ± 0.065 | 0.050 ± 0.101 | 0.211 ± 0.059 | 0.511 ± 0.426 | 0.151 ± 0.118 | 0.151 ± 0.112 |
| FLORO frozen + head (S2+S1) | 3 | 0.103 ± 0.109 | 0.444 ± 0.050 | 0.163 ± 0.102 | 0.565 ± 0.067 | 0.126 ± 0.145 | 0.077 ± 0.069 | 0.116 ± 0.122 |
| TerraMind-L frozen + head (S2 12 bands + S1) | 3 | 0.233 ± 0.098 | 0.456 ± 0.035 | 0.270 ± 0.147 | 0.478 ± 0.267 | 0.464 ± 0.381 | 0.216 ± 0.093 | 0.242 ± 0.100 |
| Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1) | 3 | n/a (no radar, empty input) | | | | | | |
| Ablation: FLORO S2+S1 with dataset_raw input scaling | 3 | 0.250 ± 0.002 [0.217–0.283] | 0.235 ± 0.007 | 0.211 ± 0.013 | 0.252 ± 0.001 | 0.976 ± 0.021 | 0.257 ± 0.001 | 0.260 ± 0.004 |
| Constant map: all mangrove | – | 0.211 | 0.106 | 0 | 0.211 | 1 | – | – |
| Constant map: all non-mangrove | – | 0 | 0.394 | 0 | – | 0 | – | – |

## Mangrove IoU per site, clean imagery

| Model | segara_anakan | kubu_raya | mimika | kendari |
|---|---|---|---|---|
| Per-pixel gradient boosting (S2+S1) | 0.411 ± 0.011 | 0.859 ± 0.000 | 0.871 ± 0.001 | 0.471 ± 0.003 |
| U-Net from scratch (S2+S1) | 0.444 ± 0.037 | 0.889 ± 0.027 | 0.901 ± 0.007 | 0.466 ± 0.040 |
| FLORO frozen + head (S2+S1) | 0.386 ± 0.021 | 0.878 ± 0.011 | 0.887 ± 0.017 | 0.507 ± 0.035 |
| FLORO frozen + head (S2 only) | 0.384 ± 0.045 | 0.822 ± 0.024 | 0.887 ± 0.006 | 0.465 ± 0.018 |
| TerraMind-L frozen + head (S2 12 bands + S1) | 0.464 ± 0.042 | 0.877 ± 0.017 | 0.890 ± 0.007 | 0.515 ± 0.041 |
| Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1) | 0.412 ± 0.022 | 0.814 ± 0.018 | 0.892 ± 0.007 | 0.470 ± 0.032 |
| Ablation: FLORO S2+S1 with dataset_raw input scaling | 0.305 ± 0.027 | 0.819 ± 0.017 | 0.815 ± 0.037 | 0.446 ± 0.077 |

Note: at Kendari about a third of the pixels labelled mangrove in GMW 2020 appear as aquaculture ponds or open land in the 2019-2021 imagery (NDVI < 0.3), so scores there measure agreement with the GMW map rather than the ability to recognise mangrove alone.

## Per seed (pooled mangrove IoU)

| Model | clean | cloud50 | cloud100 |
|---|---|---|---|
| Per-pixel gradient boosting (S2+S1) | 0.762, 0.756, 0.758 | 0.370, 0.369, 0.371 | 0.223, 0.213, 0.230 |
| Per-pixel gradient boosting, radar only (reference) | 0.250, 0.250, 0.250 | 0.250, 0.250, 0.250 | 0.250, 0.250, 0.250 |
| U-Net from scratch (S2+S1) | 0.785, 0.779, 0.815, 0.795, 0.801 | 0.470, 0.442, 0.734, 0.445, 0.416 | 0.058, 0.224, 0.239, 0.221, 0.000 |
| FLORO frozen + head (S2+S1) | 0.793, 0.772, 0.791 | 0.648, 0.630, 0.639 | 0.012, 0.073, 0.224 |
| FLORO frozen + head (S2 only) | 0.777, 0.743, 0.786 | 0.243, 0.466, 0.263 | - |
| TerraMind-L frozen + head (S2 12 bands + S1) | 0.792, 0.799, 0.817 | 0.740, 0.567, 0.742 | 0.177, 0.176, 0.345 |
| Prithvi-EO-2.0-300M frozen + head (6 S2 bands, no S1) | 0.765, 0.784, 0.786 | 0.580, 0.647, 0.601 | 0.000, 0.000, 0.000 |
| Ablation: FLORO S2+S1 with dataset_raw input scaling | 0.731, 0.686, 0.684 | 0.294, 0.468, 0.286 | 0.251, 0.251, 0.248 |
