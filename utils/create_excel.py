import pandas as pd
import itertools
import os
import numpy as np
from data import *

def create_reaction_space():

    # Generate all possible combinations
    combinations = list(itertools.product(
        reactants1.items(),
        reactants2.items(),
        catalysts.items(),
        solvents.items(),
        bases.items()
    ))


    # Set random seed for reproducibility
    np.random.seed(42)  # Using fixed seed for consistency

    # Create DataFrame
    rows = []
    selected_indices = set(np.random.choice(len(combinations), 15, replace=False))

    for idx, combo in enumerate(combinations):
        row = {
            'Reactant1_SMILES': combo[0][1],
            'Reactant2_SMILES': combo[1][1],
            'Catalyst_SMILES': combo[2][1],
            'Solvent_SMILES': combo[3][1],
            'Base_SMILES': combo[4][1],
            'Yield': np.random.randint(1, 101) if idx in selected_indices else -1  # Random yield between 1-100 for selected rows
        }
        rows.append(row)    # Create DataFrame and save to Excel
    df = pd.DataFrame(rows)
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Create the catsci_data directory in project root
    os.makedirs(os.path.join(project_root, "catsci_data"), exist_ok=True)
    # Define the output path relative to project root
    filename = "reaction_space_1.xlsx"
    path = os.path.join("catsci_data", filename)
    full_path = os.path.join(project_root, path)
    print(f"Saving to: {full_path}")
    df.to_excel(full_path, index=False)
    
    return path  # Return the full absolute path