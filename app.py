import streamlit as st
import pandas as pd
import numpy as np
import shutil
import os
import time
from pathlib import Path

# Import project utilities (assumes this app is run from repo root)
from utils.generate_data import create_npz
from utils.suggested_reactions import suggested_reactions
from main import main as predict_main

# --- Configuration ---
DATA_DIR = Path("catsci_data")
MASTER_XLSX = DATA_DIR / "combinations_dataset.xlsx"
MASKED_XLSX = DATA_DIR / "catsci_test_data_masked.xlsx"
NPZ_PREFIX_MORDRED = DATA_DIR / "catsci_data_Mordred.npz"
NPZ_PREFIX_FP = DATA_DIR / "catsci_data_morgan_fp.npz"
RESULT_0 = DATA_DIR / "result_0.csv"
RESULT_1 = DATA_DIR / "result_1.csv"
SUGGESTED_OUTPUT = DATA_DIR / "suggested_reactions.xlsx"
BACKUP_DIR = DATA_DIR / "backups"

os.makedirs(BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="CatSci YieldPredict — Active Learning UI", layout="wide")
st.title("CatSci YieldPredict — Active Learning Interface")

st.markdown(
    """
    This app runs the YieldPredict active-learning loop:
    1. Convert your reaction Excel into model `.npz` files
    2. Run the dual-DEC representation + coverage selection (`main.py`)
    3. Present suggested reactions (batch)
    4. Enter experimental yields for suggested reactions and commit them back to the master Excel
    5. Repeat
    """
)

# --- Utility functions ---

def load_master(file_path: Path):
    if not file_path.exists():
        st.error(f"Master file not found: {file_path}")
        return None
    df = pd.read_excel(file_path)
    return df


def backup_master(file_path: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = BACKUP_DIR / f"combinations_dataset_backup_{ts}.xlsx"
    shutil.copy(file_path, target)
    return target


def create_masked(master_df: pd.DataFrame, masked_path: Path):
    """
    Create a masked version where unmeasured/unknown yields are set to -1.
    Here we treat yields <= 0 or NaN as unknown.
    Known positive yields are preserved.
    """
    df = master_df.copy()
    if "Yield" not in df.columns:
        # Create Yield column if missing
        df["Yield"] = -1
    # Preserve yields > 0; mark rest as -1
    df.loc[~(df["Yield"] > 0), "Yield"] = -1
    df.to_excel(masked_path, index=False)
    return masked_path


def get_suggested_df(master_df: pd.DataFrame, indices: list):
    # Indices are expected to be integer row indices
    try:
        suggested = master_df.loc[indices].copy()
    except Exception:
        # If indices are numpy list or not matching index, try iloc
        suggested = master_df.iloc[indices].copy()
    suggested.index = list(indices)
    return suggested


# --- Sidebar controls ---
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload a master combinations Excel (optional)", type=["xlsx"]) 
use_defaults = st.sidebar.checkbox("Use repository default master file", value=(uploaded is None))

batch_size = st.sidebar.number_input("Batch size (step size)", min_value=1, max_value=100, value=15, step=1)
run_once = st.sidebar.button("Suggest next batch")

st.sidebar.markdown("---")
st.sidebar.write("Actions:")
commit_button = st.sidebar.button("Commit entered yields for suggested reactions")
download_suggested = st.sidebar.button("Download latest suggested reactions")

# --- Main flow ---
# Load or save uploaded master dataset
if uploaded is not None:
    st.info("Using uploaded master Excel file")
    master_path = DATA_DIR / "combinations_dataset_uploaded.xlsx"
    with open(master_path, "wb") as f:
        f.write(uploaded.getbuffer())
    MASTER_FILE = master_path
else:
    MASTER_FILE = MASTER_XLSX

master_df = load_master(MASTER_FILE)
if master_df is None:
    st.stop()

st.subheader("Master reaction dataset (first 10 rows)")
st.dataframe(master_df.head(10))

# Provide a manual-edit link / instructions
st.markdown(
    "If you need to edit yields, modify the master Excel and re-upload, or use the commit button after entering yields for suggested reactions below."
)

# Create masked file and npz when user clicks suggest
if run_once:
    with st.spinner("Creating masked dataset and .npz files..."):
        # backup master
        backup = backup_master(MASTER_FILE)
        st.success(f"Backup created: {backup.name}")
        masked_path = create_masked(master_df, MASKED_XLSX)
        st.write(f"Masked file written to: {masked_path}")
        # Some utility scripts expect an output file path and additional args
        # In catsci_main.create_npz(output_file, "catsci_data", 7)
        try:
            create_npz(str(masked_path), str(DATA_DIR), 7)
            st.success("Finished creating npz files.")
        except Exception as e:
            st.error(f"create_npz failed: {e}")
            st.stop()

    # Now call the active learning main
    with st.spinner("Running the active-learning engine (may take a while)..."):
        # main.py expects step_size to be set internally (default 15). We patch the variable by editing code is not feasible here,
        # so we will set environment variable or attempt to monkeypatch; however the main() in main.py uses internal step_size variable.
        # As a pragmatic approach, we will accept the default batch size in main.py and post-filter if the user requested a different batch size.
        try:
            recommended = predict_main()
        except Exception as e:
            st.error("predict_main() raised an exception: {}".format(e))
            st.stop()

    if not isinstance(recommended, (list, np.ndarray)):
        st.error(f"Unexpected return type from predict_main(): {type(recommended)}")
        st.stop()

    recommended = list(map(int, list(recommended)))
    st.success(f"Recommendation produced ({len(recommended)} indices)")

    # If user requested a different batch size than the main.py default, trim or pad
    if len(recommended) > batch_size:
        recommended = recommended[:batch_size]
    elif len(recommended) < batch_size:
        st.warning(f"Model returned {len(recommended)} suggestions; you requested {batch_size}.")

    st.session_state['last_recommendations'] = recommended

# Show last suggestions if present
if 'last_recommendations' in st.session_state:
    rec_ids = st.session_state['last_recommendations']
    st.subheader(f"Last suggested reactions ({len(rec_ids)})")

    # Get subset of master corresponding to suggested reaction IDs
    suggested_df = get_suggested_df(master_df, rec_ids)

    st.markdown("### Edit and enter yields directly")
    st.info("You can type or adjust yield values directly in the table below. Leave as -1 if not yet measured.")

    # Editable data table
    edited_df = st.data_editor(
        suggested_df,
        key="edited_suggested_df",
        num_rows="fixed",
        use_container_width=True
    )

    # Save to session for later commit
    st.session_state['edited_suggested_df'] = edited_df

    # Commit updated yields button
    if st.button("Commit updated yields to master dataset"):
        edited_df = st.session_state.get('edited_suggested_df', None)
        if edited_df is None or edited_df.empty:
            st.warning("No data found to commit.")
        else:
            # Backup master before making changes
            backup = backup_master(MASTER_FILE)
            st.info(f"Backup created before commit: {backup.name}")

            # Update yields in master_df where reaction IDs match
            for idx, row in edited_df.iterrows():
                if 'Yield' in row and pd.notna(row['Yield']) and row['Yield'] >= 0:
                    try:
                        master_df.loc[master_df['Reaction_ID'] == row['Reaction_ID'], 'Yield'] = float(row['Yield'])
                    except Exception as e:
                        st.error(f"Could not update reaction ID {row['Reaction_ID']}: {e}")

            # Save updated master dataset
            master_df.to_excel(MASTER_FILE, index=False)
            st.success("Updated yields have been committed to the master Excel file. You can now re-run 'Suggest next batch' to continue the loop.")

# Download latest suggested as Excel
if download_suggested and 'last_recommendations' in st.session_state:
    rec_ids = st.session_state['last_recommendations']
    suggested_df = get_suggested_df(master_df, rec_ids)
    suggested_df.to_excel(SUGGESTED_OUTPUT, index=False)
    with open(SUGGESTED_OUTPUT, 'rb') as f:
        st.download_button("Download suggested reactions (xlsx)", f, file_name="suggested_reactions.xlsx")

st.caption("Note: this interface runs the compute synchronously. Large datasets and DEC training may take a long time to complete.")
