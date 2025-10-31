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
    st.experimental_rerun()

# Show last suggestions if present
if 'last_recommendations' in st.session_state:
    rec_ids = st.session_state['last_recommendations']
    st.subheader(f"Last suggested reactions ({len(rec_ids)})")
    suggested_df = get_suggested_df(master_df, rec_ids)
    st.dataframe(suggested_df)

    # Allow user to edit/enter yields inline
    st.markdown("### Enter yields for suggested reactions")
    yield_inputs = {}
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("View and optionally edit reaction values; enter numeric yields for performed experiments.")
    with col2:
        st.write("Leave blank if not yet measured.")

    edited_vals = {}
    for idx, row in suggested_df.iterrows():
        default_val = row.get('Yield', -1)
        # Render a number_input for each suggested row
        val = st.number_input(f"Index {idx}", min_value=-1.0, max_value=100.0, value=float(default_val if pd.notna(default_val) else -1.0), step=1.0, key=f"yield_{idx}")
        edited_vals[idx] = val

    st.session_state['edited_yields'] = edited_vals

    if commit_button:
        # Commit updated yields back to master_df
        edited = st.session_state.get('edited_yields', {})
        if not edited:
            st.warning("No yields entered to commit.")
        else:
            # Backup master before change
            backup = backup_master(MASTER_FILE)
            st.info(f"Backup before commit: {backup.name}")
            for idx, y in edited.items():
                # Only commit if yield >= 0 (meaning user set a real value)
                if y is None:
                    continue
                try:
                    yfloat = float(y)
                except Exception:
                    continue
                if yfloat >= 0:
                    # assign by position (iloc) if index not equal
                    try:
                        master_df.loc[idx, 'Yield'] = yfloat
                    except Exception:
                        # fallback to iloc if idx is not label
                        master_df.iloc[idx, master_df.columns.get_loc('Yield')] = yfloat
            # Save master
            master_df.to_excel(MASTER_FILE, index=False)
            st.success("Committed yields to master Excel. You can re-run 'Suggest next batch' to continue the loop.")

# Download latest suggested as Excel
if download_suggested and 'last_recommendations' in st.session_state:
    rec_ids = st.session_state['last_recommendations']
    suggested_df = get_suggested_df(master_df, rec_ids)
    suggested_df.to_excel(SUGGESTED_OUTPUT, index=False)
    with open(SUGGESTED_OUTPUT, 'rb') as f:
        st.download_button("Download suggested reactions (xlsx)", f, file_name="suggested_reactions.xlsx")

# Small status / housekeeping
st.markdown("---")
st.write("Model intermediate outputs (if present):")
cols = st.columns(2)
with cols[0]:
    if RESULT_0.exists():
        st.write(f"{RESULT_0.name} exists")
        if st.button("Show result_0.csv"):
            df0 = pd.read_csv(RESULT_0, header=None)
            st.dataframe(df0.head(20))
    else:
        st.write("result_0.csv not present yet")
with cols[1]:
    if RESULT_1.exists():
        st.write(f"{RESULT_1.name} exists")
        if st.button("Show result_1.csv"):
            df1 = pd.read_csv(RESULT_1, header=None)
            st.dataframe(df1.head(20))
    else:
        st.write("result_1.csv not present yet")

st.caption("Note: this interface runs the compute synchronously. Large datasets and DEC training may take time to complete.")
