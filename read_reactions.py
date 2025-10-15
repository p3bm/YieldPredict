import pandas as pd

# Read the Excel file
# df = pd.read_excel('dataset/SM.xlsx')
df = pd.read_excel('catsci_data/reaction_space.xlsx')

# Print the first few rows to see the structure
print("Dataset shape:", df.shape)
print("\nFirst few reactions:")
print(df.head())

# If indices are provided, show specific reactions
# indices = [580, 1660, 498, 2845, 2313]  # example indices from your output
indices = [94, 188, 221, 306, 328, 28, 30, 52, 116, 130, 225, 305, 338, 21, 41, 71, 103, 283, 335, 6, 7, 29, 35, 72, 108]
print("\nSelected reactions:")
for idx in indices:
    if idx < len(df):
        print(f"\nReaction {idx}:")
        print(df.iloc[idx])
