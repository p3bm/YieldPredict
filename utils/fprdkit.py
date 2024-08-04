from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
from rdkit.Avalon import pyAvalonTools
import numpy as np

def encode_dataset(sheet, num_columns_of_sheet):
    num_rows = sheet.nrows
    yields = []
    dim_molecule = 512
    num_dim = dim_molecule * (num_columns_of_sheet - 1)
    flattened_encoding_of_dataset = np.zeros((num_rows - 1, num_dim))
    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)
        yields.append(row[num_columns_of_sheet-1])
        
        for j in range(num_columns_of_sheet-1):
            smiles = row[j]
            mol = Chem.MolFromSmiles(smiles)
            encoding_of_smiles = pyAvalonTools.GetAvalonFP(mol)
            flattened_encoding_of_dataset[row_idx - 1, j * dim_molecule:(j + 1) * dim_molecule] = encoding_of_smiles

    return flattened_encoding_of_dataset, np.array(yields)