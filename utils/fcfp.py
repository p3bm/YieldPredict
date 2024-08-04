import numpy as np
from xenonpy.descriptor import Fingerprints

def encode_dataset(sheet, num_columns_of_sheet):
    fp = Fingerprints(featurizers='FCFP', input_type='smiles')
    num_rows = sheet.nrows
    # num_dim = dim_molecule * (num_columns_of_sheet - 1)
    # flattened_encoding_of_dataset = np.zeros((num_rows-1, num_dim))
    yields = []
    
    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)
        yields.append(row[num_columns_of_sheet-1])
        
        encoding_of_smiles = fp.transform(row[:-1]).to_numpy().reshape(1, -1).squeeze()
        # print(encoding_of_smiles.shape)
        if row_idx == 1:
            flattened_encoding_of_dataset = np.zeros((num_rows-1, len(encoding_of_smiles)))
        flattened_encoding_of_dataset[row_idx-1, :] = encoding_of_smiles
       
    return flattened_encoding_of_dataset, np.array(yields)