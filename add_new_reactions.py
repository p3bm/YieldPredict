import pandas as pd
import json
import os
from datetime import datetime

def load_reaction_space():
    """Load the current reaction space"""
    return pd.read_excel("catsci_data/reaction_space.xlsx")

def load_suggested_reactions():
    """Load the suggested reactions"""
    return pd.read_excel("catsci_data/suggested_reactions.xlsx")

def load_existing_reactions():
    """Load existing reactions from storage"""
    if os.path.exists("catsci_data/reaction_history.json"):
        with open("catsci_data/reaction_history.json", "r") as f:
            return json.load(f)
    return {"iterations": []}

def save_reactions(reactions_data):
    """Save reactions to storage"""
    with open("catsci_data/reaction_history.json", "w") as f:
        json.dump(reactions_data, f, indent=2)

def add_new_reaction_results():
    """Add new reaction results interactively"""
    df = load_suggested_reactions()
    reactions_data = load_existing_reactions()
    
    print("\nCurrent suggested reactions:")
    for idx, row in df.iterrows():
        print(f"\nReaction {idx + 1}:")
        print(f"Reactant1: {row['Reactant1_SMILES'][:30]}...")
        print(f"Reactant2: {row['Reactant2_SMILES'][:30]}...")
        print(f"Catalyst: {row['Catalyst_SMILES'][:30]}...")
        print(f"Solvent: {row['Solvent_SMILES']}")
        print(f"Base: {row['Base_SMILES']}")
        
        while True:
            try:
                yield_input = input(f"\nEnter yield for reaction {idx + 1} (-1 to skip, or yield value 0-100): ")
                yield_value = float(yield_input)
                if yield_value == -1:
                    print("Skipping this reaction...")
                    break
                if 0 <= yield_value <= 100:
                    new_reaction = {
                        'Reactant1_SMILES': row['Reactant1_SMILES'],
                        'Reactant2_SMILES': row['Reactant2_SMILES'],
                        'Catalyst_SMILES': row['Catalyst_SMILES'],
                        'Solvent_SMILES': row['Solvent_SMILES'],
                        'Base_SMILES': row['Base_SMILES'],
                        'Yield': yield_value
                    }
                    
                    # Add to history with metadata
                    iteration_data = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reaction": new_reaction
                    }
                    reactions_data["iterations"].append(iteration_data)
                    break
                else:
                    print("Yield must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
    
    # Save updated reactions
    save_reactions(reactions_data)
    print("\nReactions saved successfully!")
    
    # Show summary
    print("\nSummary of all reactions:")
    for idx, iteration in enumerate(reactions_data["iterations"]):
        print(f"\nReaction {idx + 1} (Added: {iteration['date']}):")
        print(f"Yield: {iteration['reaction']['Yield']}")

def update_reaction_space():
    """Update the reaction space Excel file with new reactions"""
    reactions_data = load_existing_reactions()
    df = load_reaction_space()
    
    # Get all reactions from history
    all_reactions = [iteration["reaction"] for iteration in reactions_data["iterations"]]
    
    # Update reaction space
    for reaction in all_reactions:
        mask = (
            (df['Reactant1_SMILES'] == reaction['Reactant1_SMILES']) &
            (df['Reactant2_SMILES'] == reaction['Reactant2_SMILES']) &
            (df['Catalyst_SMILES'] == reaction['Catalyst_SMILES']) &
            (df['Solvent_SMILES'] == reaction['Solvent_SMILES']) &
            (df['Base_SMILES'] == reaction['Base_SMILES'])
        )
        df.loc[mask, 'Yield'] = reaction['Yield']
    
    # Save updated reaction space
    df.to_excel("catsci_data/reaction_space.xlsx", index=False)
    print("\nReaction space updated successfully!")
    print(f"Total non-negative yields: {(df['Yield'] > -1).sum()}")

def main():
    while True:
        print("\n=== Reaction Management System ===")
        print("1. Add new reaction results")
        print("2. Update reaction space")
        print("3. Show reaction history")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            add_new_reaction_results()
        elif choice == "2":
            update_reaction_space()
        elif choice == "3":
            reactions_data = load_existing_reactions()
            print("\nReaction History:")
            for idx, iteration in enumerate(reactions_data["iterations"]):
                print(f"\nReaction {idx + 1} (Added: {iteration['date']}):")
                print(f"Yield: {iteration['reaction']['Yield']}")
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
