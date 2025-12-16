import json
import os
import time

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# --- CONFIGURATION ---
# The CSV containing the recipes to evaluate (Read-Only)
DATA_FILE = "Recipes evaluation - evaluation.csv"
ANNOTATORS = ["Chef_Mario", "Chef_Luigi", "Chef_Peach"]

st.set_page_config(layout="wide", page_title="Recipe Eval (Google Sheets)")

# ==========================================
# 1. LOGIN SYSTEM
# ==========================================
if "annotator_name" not in st.session_state:
    st.title("🔐 Annotator Login")
    st.markdown("Please identify yourself to sync your personal progress.")

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_name = st.selectbox("Select your name:", ANNOTATORS, index=None)

        if st.button("🚀 Start Evaluation", type="primary", use_container_width=True):
            if selected_name:
                st.session_state.annotator_name = selected_name
                st.rerun()
            else:
                st.error("⚠️ You must select a name.")
    st.stop()

# Current User is set
current_user = st.session_state.annotator_name


# ==========================================
# 2. DATA LOADING & GOOGLE SHEETS
# ==========================================
@st.cache_data
def load_source_data():
    """Loads the source recipes from the local CSV."""
    if not os.path.exists(DATA_FILE):
        return []

    df = pl.read_csv(DATA_FILE)
    np.random.seed(42)
    is_mixed_A = np.random.rand(len(df)) < 0.5

    # Persist mapping if needed
    if not os.path.exists("mapping_reference.csv"):
        mapping = [
            {"id": i, "Mixed_is": "A" if b else "B"} for i, b in enumerate(is_mixed_A)
        ]
        pl.DataFrame(mapping).write_csv("mapping_reference.csv")

    prepared_data = []
    # Adjust column names as needed based on your file
    rows = df.select(
        "title", "output_Qwen3-4B-Cross-Entropy", "output_Qwen3-4B-Mixed"
    ).iter_rows()

    for i, (row, mixed) in enumerate(zip(rows, is_mixed_A)):

        def parse(val):
            try:
                if isinstance(val, (dict, list)):
                    return val
                return json.loads(val)
            except:
                return {"ingredients": [], "instructions": [str(val)]}

        ce = parse(row[1])
        mixed_out = parse(row[2])

        prepared_data.append(
            {
                "ID": i,
                "title": row[0],
                "A": mixed_out if mixed else ce,
                "B": ce if mixed else mixed_out,
            }
        )
    return prepared_data


data = load_source_data()

# Connect to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)


def get_google_sheet_data():
    """Reads the Google Sheet. Returns empty DF if sheet is empty."""
    df = conn.read(ttl=0)
    return df
    try:
        # ttl=0 means do not cache, always get fresh data
        df = conn.read(ttl=0)
        return df
    except Exception:
        return pd.DataFrame()


def save_to_google_sheet(new_record):
    """Reads current sheet, appends new row, writes back."""
    df_existing = get_google_sheet_data()
    df_new = pd.DataFrame([new_record])

    # Concatenate
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    conn.update(data=df_combined)


# ==========================================
# 3. PROGRESS TRACKING (User Specific)
# ==========================================
# Load the sheet data to check progress
df_results = get_google_sheet_data()

# Filter: Which IDs has THIS SPECIFIC USER completed?
completed_ids = set()
if (
    not df_results.empty
    and "sample_id" in df_results.columns
    and "annotator" in df_results.columns
):
    # We filter only rows where 'annotator' == current_user
    user_rows = df_results[df_results["annotator"] == current_user]
    completed_ids = set(user_rows["sample_id"].unique())

if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def go_next():
    if st.session_state.current_idx < len(data) - 1:
        st.session_state.current_idx += 1


# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.header(f"👤 {current_user}")
    if st.button("Log out", key="logout"):
        del st.session_state.annotator_name
        st.rerun()

    st.divider()

    # Progress Bar
    completed_count = len(completed_ids)
    total = len(data)
    st.progress(completed_count / total if total > 0 else 0)
    st.write(f"**{completed_count}/{total}** completed")

    st.markdown("---")

    # Navigation Dropdown
    # We mark with ✅ only if THIS user has done it
    options = [
        f"{d['ID']}: {d['title']} {'✅' if d['ID'] in completed_ids else ''}"
        for d in data
    ]
    selected_opt = st.selectbox(
        "Navigate:", options, index=st.session_state.current_idx
    )
    new_idx = int(selected_opt.split(":")[0])
    if new_idx != st.session_state.current_idx:
        st.session_state.current_idx = new_idx
        st.rerun()

    if st.button("⏭️ Find My Next Pending"):
        for i in range(len(data)):
            if data[i]["ID"] not in completed_ids:
                st.session_state.current_idx = i
                st.rerun()
        st.success("You have completed all samples!")

# ==========================================
# 5. MAIN UI
# ==========================================
current_pair = data[st.session_state.current_idx]
sample_id = current_pair["ID"]

st.title("👨‍🍳 Recipe Evaluation (ACL)")

# Status Banner
if sample_id in completed_ids:
    st.warning(
        f"⚠️ You ({current_user}) have already evaluated Sample #{sample_id}. Submitting again will create a duplicate."
    )

st.markdown(f"## 🍽️ *{current_pair['title']}*")


# --- Render Recipes ---
def render(r):
    st.markdown("#### Ingredients")
    st.write(r.get("ingredients", []))
    st.markdown("#### Instructions")
    inst = r.get("instructions", [])
    if isinstance(inst, list):
        for i, s in enumerate(inst, 1):
            st.markdown(f"**{i}.** {s}")
    else:
        st.write(inst)


c1, c2 = st.columns(2)
with c1:
    st.info("Versione A")
    render(current_pair["A"])
with c2:
    st.success("Versione B")
    render(current_pair["B"])

st.divider()

# ==========================================
# 6. FORM
# ==========================================
# KEY includes sample_id so form resets on navigation
with st.form(key=f"form_{sample_id}"):
    st.subheader("Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        p_ing = st.radio(
            "Ingredients Preference",
            ["A", "B", "Tie"],
            horizontal=True,
            key=f"pi_{sample_id}",
            index=None,
        )
        p_num = st.radio(
            "Numbers/Qty Preference",
            ["A", "B", "Tie"],
            horizontal=True,
            key=f"pn_{sample_id}",
            index=None,
        )
    with col_b:
        p_proc = st.radio(
            "Procedure Preference",
            ["A", "B", "Tie"],
            horizontal=True,
            key=f"pp_{sample_id}",
            index=None,
        )
        p_all = st.radio(
            "🏆 Overall Preference",
            ["A", "B", "Tie"],
            horizontal=True,
            key=f"pall_{sample_id}",
            index=None,
        )

    st.subheader("Expert Check")
    ec1, ec2 = st.columns(2)
    err_opts = [
        "Missing Ingredient",
        "Hallucination",
        "Bad Qty",
        "Bad Time",
        "Bad Temperature",
        "Step Mismatch",
        "Safety Issue",
    ]

    with ec1:
        st.markdown("**Version A**")
        ac = st.selectbox("Cookable?", ["Yes", "Maybe", "No"], key=f"ac_{sample_id}")
        at = st.slider("Trust (1-5)", 1, 5, 3, key=f"at_{sample_id}")
        ae = st.multiselect("Errors A", err_opts, key=f"ae_{sample_id}")

    with ec2:
        st.markdown("**Version B**")
        bc = st.selectbox("Cookable?", ["Yes", "Maybe", "No"], key=f"bc_{sample_id}")
        bt = st.slider("Trust (1-5)", 1, 5, 3, key=f"bt_{sample_id}")
        be = st.multiselect("Errors B", err_opts, key=f"be_{sample_id}")

    notes = st.text_area("Notes", key=f"nt_{sample_id}")

    # SUBMIT
    submitted = st.form_submit_button("☁️ Save to Google Drive", type="primary")

    if submitted:
        if not all([p_ing, p_num, p_proc, p_all]):
            st.error("Please fill all comparison fields.")
        else:
            # Construct Record
            record = {
                "annotator": current_user,  # <--- VITAL for IAA
                "sample_id": sample_id,
                "recipe_title": current_pair["title"],
                "pref_ingredients": p_ing,
                "pref_numbers": p_num,
                "pref_procedure": p_proc,
                "pref_overall": p_all,
                "A_cookable": ac,
                "A_trust": at,
                "A_errors": ";".join(ae),
                "B_cookable": bc,
                "B_trust": bt,
                "B_errors": ";".join(be),
                "notes": notes,
                "timestamp": str(pd.Timestamp.now()),
            }

            with st.spinner("Saving to Google Sheets..."):
                save_to_google_sheet(record)

            st.success("Saved! Moving to next...")
            time.sleep(1)  # Brief pause to show success message
            go_next()
            st.rerun()

# Navigation Footer
c_prev, c_next = st.columns([1, 1])
with c_prev:
    if st.button("⬅️ Prev"):
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
with c_next:
    if st.button("Next ➡️"):
        go_next()
        st.rerun()
