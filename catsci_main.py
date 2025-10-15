from main import main as predict_main  # Rename the imported main function
from utils.generate_data import create_npz
from utils.create_excel import create_reaction_space
from utils.suggested_reactions import suggested_reactions
import pandas as pd
import numpy as np

# def mask_yield_column_by_index(input_file, output_file, yield_column="Yield", keep_n=15):
#     test_dataframe = pd.read_excel(input_file)
#     # Make a copy to avoid modifying the original
#     masked_test_df = test_dataframe.copy()
#     print(masked_test_df.columns)
    
#      # Get indices where Yield < 50
#     eligible_indices = masked_test_df.index[
#         (masked_test_df[yield_column] > 0) & (masked_test_df[yield_column] < 30)
#     ].tolist()
#     if len(eligible_indices) < keep_n:
#         raise ValueError(f"Not enough rows with {yield_column} < 50 to select {keep_n} indices.")

#     # np.random.seed(42)
#     keep_indices = np.random.choice(eligible_indices, size=keep_n, replace=False)


#     # Set all yields to -1 except the selected indices
#     masked_test_df.loc[~masked_test_df.index.isin(keep_indices), "Yield"] = -1

#     # Save the masked file
#     output_file = "catsci_data/catsci_test_data_masked.xlsx"
#     masked_test_df.to_excel(output_file, index=False)
#     print(f"Masked yields saved to {output_file}")

def main():

    file_path = "catsci_data/combinations_dataset.xlsx"
    # mask_yield_column_by_index(input_file= file_path, 
    #                   output_file="catsci_data/catsci_test_data_masked.xlsx",)
    output_file = "catsci_data/catsci_test_data_masked.xlsx"
    create_npz(output_file, "catsci_data", 7)
    print(f"Finished creating npz file.")
    recomended_reaction_ids = predict_main()  # Use the renamed function
    print(f"Recommended reaction IDs: {recomended_reaction_ids}")

    suggested_reactions(file_path= file_path, 
                        recommended_reaction_ids= recomended_reaction_ids)
if __name__ == "__main__":
    main()
