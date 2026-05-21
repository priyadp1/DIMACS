import pacmap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import pandas as pd

current = Path(__file__).resolve()
while current.name != "DIMACS":
    current = current.parent
BASEDIR = current
tgb_dir = BASEDIR / "TGB_Variables_Feature_Importance"
no_tgb_dir = BASEDIR / "tmp_splits_no_tgb"
pacmap_output_dir = BASEDIR / "pacmap_embeddings"
os.makedirs(pacmap_output_dir, exist_ok=True)
parameters = ["nest40_depth1", "nest40_depth2", "nest40_depth3", "nest100_depth1", "nest100_depth2", "nest100_depth3", "nest200_depth1", "nest200_depth2", "nest200_depth3"]
datasets = ["bike", "compas", "breast_cancer", "spambase"]

for i in datasets:
    no_tgb_X_train = no_tgb_dir / i / "X_train.csv"
    no_tgb_X_test  = no_tgb_dir / i / "X_test.csv"
    no_tgb_y_train = no_tgb_dir / i / "y_train.csv"
    no_tgb_y_test  = no_tgb_dir / i / "y_test.csv"
    if not no_tgb_X_train.exists():
        print(f"Skipping {i}: missing no-TGB files")
        continue
    df_no_X_train = pd.read_csv(no_tgb_X_train).values
    df_no_X_test  = pd.read_csv(no_tgb_X_test).values
    df_no_y_train = pd.read_csv(no_tgb_y_train).values.ravel()
    df_no_y_test  = pd.read_csv(no_tgb_y_test).values.ravel()
    scaler_no = StandardScaler()
    X_no_combined = scaler_no.fit_transform(np.vstack([df_no_X_train, df_no_X_test]))
    y_no_combined = np.concatenate([df_no_y_train, df_no_y_test])
    emb_no = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_no_embedded = emb_no.fit_transform(X_no_combined)
    for j in parameters:
        tgb_X_train = tgb_dir / i / j / "X_train_guessed.csv"
        tgb_X_test  = tgb_dir / i / j / "X_test_guessed.csv"
        tgb_y_train = tgb_dir / i / j / "y_train.csv"
        tgb_y_test  = tgb_dir / i / j / "y_test.csv"
        if not tgb_X_train.exists():
            print(f"Skipping {i}/{j}: missing TGB files")
            continue

        df_tgb_X_train = pd.read_csv(tgb_X_train).values
        df_tgb_X_test  = pd.read_csv(tgb_X_test).values
        df_tgb_y_train = pd.read_csv(tgb_y_train).values.ravel()
        df_tgb_y_test  = pd.read_csv(tgb_y_test).values.ravel()

        scaler_tgb = StandardScaler()
        X_tgb_combined = scaler_tgb.fit_transform(np.vstack([df_tgb_X_train, df_tgb_X_test]))
        y_tgb_combined = np.concatenate([df_tgb_y_train, df_tgb_y_test])
        emb_tgb = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        X_tgb_embedded = emb_tgb.fit_transform(X_tgb_combined)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, X_emb, y, title in [
            (axes[0], X_tgb_embedded, y_tgb_combined, f"TGB ({j})"),
            (axes[1], X_no_embedded,  y_no_combined,  "No TGB (original features)"),
        ]:
            for label, color in [(0, "steelblue"), (1, "tomato")]:
                mask = y == label
                ax.scatter(X_emb[mask, 0], X_emb[mask, 1], c=color, label=f"class {label}", alpha=0.5, s=10)
            ax.set_title(title)
            ax.legend()

        fig.suptitle(f"PaCMAP: {i} — {j}", fontsize=14)
        plt.tight_layout()
        out_path = pacmap_output_dir / f"{i}_{j}_pacmap.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")
