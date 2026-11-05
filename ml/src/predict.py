from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "random_forest_v1.joblib"


def predict_ache_activity(smiles):
    # Convert SMILES into a molecular structure
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    # Generate the same fingerprints used during training
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True
    )

    fingerprint = generator.GetFingerprintAsNumPy(mol)
    X = np.array([fingerprint])

    # Load trained Random Forest model
    model = joblib.load(MODEL_PATH)

    # Predict pIC50
    predicted_pic50 = float(model.predict(X)[0])

    # Convert pIC50 into IC50 (nM)
    predicted_ic50 = 10 ** (9 - predicted_pic50)

    return {
        "predicted_pIC50": predicted_pic50,
        "predicted_IC50_nM": predicted_ic50,
        "prediction_status": "Exploratory - not experimentally validated"
    }
