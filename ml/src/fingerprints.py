import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def smiles_to_fingerprints(smiles_list):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True
    )

    fingerprints = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = generator.GetFingerprintAsNumPy(mol)
        fingerprints.append(fp)

    return np.array(fingerprints)
