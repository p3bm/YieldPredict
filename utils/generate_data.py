import os
import numpy as np
import pandas
import pandas as pd
import xlrd2
import random

# Import from the same directory
from . import utils
from . import mdr_descriptors
from . import morgan_fingerprint as mfp

def create_npz(file_name, dataset_name, columns):
    # strs = input("xlsx filename:")
    # type = input("dataset name:")
    # num_columns_of_sheet = int(input("columns num(include yield):"))
    strs = file_name
    type = dataset_name
    num_columns_of_sheet = columns
    reprs = ["morgan_fp", "Mordred"]
    for representation in reprs:
        # print("representation:" + representation)

        excel = xlrd2.open_workbook(strs)
        sheet = excel.sheet_by_index(0)
        
        if representation == "morgan_fp":
            flattened_encoding_of_dataset, yields = mfp.encode_dataset(sheet, num_columns_of_sheet, 2048)
        elif representation == "Mordred":
            flattened_encoding_of_dataset, yields = mdr_descriptors.encode_dataset(sheet, num_columns_of_sheet)
        '''
        elif representation == "rdkit":
            flattened_encoding_of_dataset, yields = fprdkit.encode_dataset(sheet, num_columns_of_sheet)
        elif representation == "rxnfp":
            flattened_encoding_of_dataset, yields = myrxnfp.encode_dataset(sheet, num_columns_of_sheet, 5)
        elif representation == 'atompair':
            flattened_encoding_of_dataset, yields = ap.encode_dataset(sheet, num_columns_of_sheet)
            print(flattened_encoding_of_dataset.shape)
        elif representation == 'torsion':
            flattened_encoding_of_dataset, yields = tfp.encode_dataset(sheet, num_columns_of_sheet)
        elif representation == 'ECFP':
            flattened_encoding_of_dataset, yields = ECFP.encode_dataset(sheet, num_columns_of_sheet)
        elif representation == 'FCFP':
            flattened_encoding_of_dataset, yields = FCFP.encode_dataset(sheet, num_columns_of_sheet)
        '''
        # np.savez(type + '_' + representation + ".npz", train_data=flattened_encoding_of_dataset, train_labels=yields)
        # print("\n")
        output_dir = os.path.dirname(strs)
        output_file = os.path.join(output_dir, f"{type}_{representation}.npz")
        np.savez(output_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
        print(f"Saved {output_file}")

# if __name__ == "__main__":
#     create_npz(file_name="catsci_data/reaction_space.xlsx", 
#                dataset_name="catsci_suzuki", 
#                columns=6)
