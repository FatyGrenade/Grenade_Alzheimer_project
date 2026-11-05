from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ache_ml_ready.csv"
MODEL_DIR = BASE_DIR / "models"


def main():
    # Load curated ChEMBL dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["canonical_smiles", "pIC50"]).copy()

    df["mol"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).copy()

    # Generate molecular fingerprints
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True
    )

    X = np.array([
        generator.GetFingerprintAsNumPy(mol)
        for mol in df["mol"]
    ])

    y = df["pIC50"].to_numpy()

    # Use canonical structures for grouping
    groups = df["mol"].apply(
        lambda mol: Chem.MolToSmiles(mol, canonical=True)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42
        )
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")

        filename = (
            "random_forest_retrained.joblib"
            if name == "Random Forest"
            else "xgboost_retrained.joblib"
        )

        joblib.dump(model, MODEL_DIR / filename)


if __name__ == "__main__":
    main()
