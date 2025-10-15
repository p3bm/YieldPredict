import pandas as pd
import os

# After performing these reactions in the lab and getting yields, 
# you can add them to known_reactions in create_excel.ipynb like this:

# '''
# # Example of how to add new reactions to known_reactions:
# for idx in new_reaction_indices:
#     new_reaction = {
#         "Reactant1_SMILES": df.iloc[idx]["Reactant1_SMILES"],
#         "Reactant2_SMILES": df.iloc[idx]["Reactant2_SMILES"],
#         "Catalyst_SMILES": df.iloc[idx]["Catalyst_SMILES"],
#         "Solvent_SMILES": df.iloc[idx]["Solvent_SMILES"],
#         "Base_SMILES": df.iloc[idx]["Base_SMILES"],
#         "Yield": 0  # Replace with actual yield from lab experiment
#     }
#     known_reactions.append(new_reaction)
# '''




def suggested_reactions(file_path, recommended_reaction_ids):
    """
    This function is a placeholder for any additional processing or analysis
    you might want to perform on the suggested reactions.
    """
    
    # Load the reaction space
    df = pd.read_excel(file_path)

    # Get the suggested reactions
    suggested_reactions = df.iloc[recommended_reaction_ids]
    # print("\nSuggested reactions:")
    # print(suggested_reactions[['Reactant1_SMILES', 'Reactant2_SMILES', 'Catalyst_SMILES', 'Solvent_SMILES', 'Base_SMILES']])
    # Save reactions to Excel for lab reference
    output_dir = os.path.dirname(file_path)
    suggested_file_name = output_dir + "/suggested_reactions.xlsx"
    suggested_reactions.to_excel(suggested_file_name, index=True)
    print("\nSaved suggested reactions to 'catsci_data/suggested_reactions.xlsx'")