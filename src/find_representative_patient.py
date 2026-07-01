#!/usr/bin/env python3
"""Find a representative patient with multi-lobar tumor involvement
for the anatomical overlay figure."""

import pandas as pd

df = pd.read_csv("outputs/multimodal_lobewise/merged_features_with_metadata.csv")

en_cols = ["T1_frontal_en_ratio", "T1_temporal_en_ratio",
           "T1_parietal_en_ratio", "T1_occipital_en_ratio"]
nc_cols = ["T1_frontal_nc_ratio", "T1_temporal_nc_ratio",
           "T1_parietal_nc_ratio", "T1_occipital_nc_ratio"]
ed_cols = ["T1_frontal_ed_ratio", "T1_temporal_ed_ratio",
           "T1_parietal_ed_ratio", "T1_occipital_ed_ratio"]

df["en_lobes"] = (df[en_cols] > 0.001).sum(axis=1)
df["nc_lobes"] = (df[nc_cols] > 0.001).sum(axis=1)
df["ed_lobes"] = (df[ed_cols] > 0.001).sum(axis=1)

# Need: visible ET, NC, ED with multi-lobar involvement (>=2 lobes for ET)
mask = (df["en_lobes"] >= 2) & (df["nc_lobes"] >= 1) & (df["ed_lobes"] >= 1)
candidates = df[mask].sort_values(["en_lobes", "nc_lobes", "ed_lobes"],
                                   ascending=False)

lobe_names = ["frontal", "temporal", "parietal", "occipital"]

print("Top 10 candidates:\n")
for idx, row in candidates.head(10).iterrows():
    pid = row["patient_id"]
    print(f"{pid}: en={row['en_lobes']}, nc={row['nc_lobes']}, ed={row['ed_lobes']}")
    for sub, cols in [("ET", en_cols), ("NC", nc_cols), ("ED", ed_cols)]:
        vals = [f"{l}={row[c]:.4f}" for l, c in zip(lobe_names, cols)]
        print(f"  {sub}: " + ", ".join(vals))
    print()

# Check if candidate exists in UCSF data dir
import os
base = "UCSF/DATA-IMAGE-STRUCTURAL"
pid_short = candidates.iloc[0]["patient_id"]
path = os.path.join(base, pid_short)
print(f"Best candidate: {pid_short}")
print(f"Exists in image dir: {os.path.isdir(path)}")

# Also check segmentation
seg_path = f"UCSF/DATA-AUTOMATED-SEGMENT/{pid_short}_tumor_segmentation.nii.gz"
print(f"Segmentation exists: {os.path.isfile(os.path.join('..', seg_path))}")
