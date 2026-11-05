from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "ache_ml_ready.csv"
REPORTS = BASE / "reports"

# 1. Load data
df = pd.read_csv(DATA)
df = df.dropna(subset=["canonical_smiles", "pIC50"]).copy()

df["mol"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
df = df.dropna(subset=["mol"]).copy()

print("Valid molecules:", len(df))


# 2. Extract Bemis-Murcko scaffolds
def get_scaffold(mol):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    if scaffold:
        return scaffold

    # Acyclic molecules have no Murcko scaffold.
    # Group each distinct acyclic structure separately.
    return "ACYCLIC:" + Chem.MolToSmiles(mol, canonical=True)


df["scaffold"] = df["mol"].apply(get_scaffold)

print("Unique scaffold groups:", df["scaffold"].nunique())


# 3. Generate Morgan fingerprints
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


# 4. Scaffold-based train/test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["scaffold"])
)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

train_scaffolds = set(df.iloc[train_idx]["scaffold"])
test_scaffolds = set(df.iloc[test_idx]["scaffold"])

overlap = train_scaffolds.intersection(test_scaffolds)

print("Training molecules:", len(train_idx))
print("Testing molecules:", len(test_idx))
print("Overlapping scaffold groups:", len(overlap))

assert len(overlap) == 0, "Scaffold leakage detected!"


# 5. Train and evaluate
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

results = []

for name, model in models.items():
    print(f"\nTraining {name} with scaffold split...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{name} MAE: {mae:.4f}")
    print(f"{name} R2: {r2:.4f}")

    results.append({
        "model": name,
        "split": "Bemis-Murcko scaffold",
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "MAE": mae,
        "R2": r2
    })


# 6. Save results without overwriting original model
REPORTS.mkdir(parents=True, exist_ok=True)

output = REPORTS / "scaffold_split_metrics.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("\nResults saved to:", output)
