import os
import numpy as np
import pandas
import pandas as pd
import xlrd2

import utils
import random

import mdr_descriptors
import morgan_fingerprint as mfp

def main():
    strs = input("xlsx filename")
    type = input("dataset name")
    num_columns_of_sheet = int(input("columns num(include yield)"))
    reprs = ["morgan_fp", "Mordred"]
    for representation in reprs:
        print("representation:" + representation)

        excel = xlrd2.open_workbook(strs)
        sheet = excel.sheet_by_index(0)
        
        if args.representation == "morgan_fp":
            flattened_encoding_of_dataset, yields = mfp.encode_dataset(sheet, num_columns_of_sheet, 2048)
        elif args.representation == "Mordred":
            flattened_encoding_of_dataset, yields = mdr_descriptors.encode_dataset(sheet, num_columns_of_sheet)
        '''
        elif args.representation == "rdkit":
            flattened_encoding_of_dataset, yields = fprdkit.encode_dataset(sheet, num_columns_of_sheet)
        elif args.representation == "rxnfp":
            flattened_encoding_of_dataset, yields = myrxnfp.encode_dataset(sheet, num_columns_of_sheet, 5)
        elif args.representation == 'atompair':
            flattened_encoding_of_dataset, yields = ap.encode_dataset(sheet, num_columns_of_sheet)
            print(flattened_encoding_of_dataset.shape)
        elif args.representation == 'torsion':
            flattened_encoding_of_dataset, yields = tfp.encode_dataset(sheet, num_columns_of_sheet)
        elif args.representation == 'ECFP':
            flattened_encoding_of_dataset, yields = ECFP.encode_dataset(sheet, num_columns_of_sheet)
        elif args.representation == 'FCFP':
            flattened_encoding_of_dataset, yields = FCFP.encode_dataset(sheet, num_columns_of_sheet)
        '''
        np.savez(type + '_' + representation + ".npz", train_data=flattened_encoding_of_dataset, train_labels=yields)
        print("\n")

if __name__ == "__main__":
    main()
