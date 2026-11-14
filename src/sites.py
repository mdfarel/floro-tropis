"""Pilot sites: 0.25 degree boxes picked from Global Mangrove Watch v3 (2020).

Train and test sites are separate places, so the test measures transfer to unseen locations.
bbox = (west, south, east, north) in degrees.
"""

SITES = {
    # train
    "mahakam": {"bbox": (117.325, -0.775, 117.575, -0.525), "split": "train", "region": "Kalimantan Timur"},
    "sembilang": {"bbox": (104.375, -2.075, 104.625, -1.825), "split": "train", "region": "Sumatera Selatan"},
    "bintuni": {"bbox": (133.675, -2.325, 133.925, -2.075), "split": "train", "region": "Papua Barat"},
    "langsa": {"bbox": (97.925, 4.425, 98.175, 4.675), "split": "train", "region": "Aceh"},
    # test (unseen sites)
    "segara_anakan": {"bbox": (108.775, -7.855, 109.025, -7.605), "split": "test", "region": "Jawa Tengah"},
    "kubu_raya": {"bbox": (109.275, -0.625, 109.525, -0.375), "split": "test", "region": "Kalimantan Barat"},
    "mimika": {"bbox": (136.675, -4.925, 136.925, -4.675), "split": "test", "region": "Papua Tengah"},
    "kendari": {"bbox": (122.475, -4.425, 122.725, -4.175), "split": "test", "region": "Sulawesi Tenggara"},
}

S2_BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
# FLORO band names for the same order
FLORO_OPTICAL = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["vv", "vh"]
