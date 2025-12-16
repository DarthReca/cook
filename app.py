import json
import os

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# --- CONFIGURATION ---
RESULTS_FILE = "valutazioni_chefs_detailed.csv"
DATA_FILE = "Recipes evaluation - evaluation.csv"

st.set_page_config(layout="wide", page_title="Recipe Eval (ACL)")


# --- 1. DATA LOADING ---
@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_FILE):
        return []

    df = pl.read_csv(DATA_FILE)

    np.random.seed(42)
    is_mixed_A = np.random.rand(len(df)) < 0.5

    # Persist mapping for reproducibility
    if not os.path.exists("mapping_reference.csv"):
        mapping = [
            {"id": i, "Mixed_is": "A" if b else "B", "CE_is": "B" if b else "A"}
            for i, b in enumerate(is_mixed_A)
        ]
        pl.DataFrame(mapping).write_csv("mapping_reference.csv")

    prepared_data = []
    # Adjust column selection if your CSV headers differ
    rows = df.select(
        "title", "output_Qwen3-4B-Cross-Entropy", "output_Qwen3-4B-Mixed"
    ).iter_rows()

    for i, (row, mixed) in enumerate(zip(rows, is_mixed_A)):
        title = row[0]
        raw_ce = row[1]
        raw_mixed = row[2]

        def parse_field(val):
            try:
                if isinstance(val, (dict, list)):
                    return val
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return {"ingredients": ["Parse Error"], "instructions": [str(val)]}

        ce = parse_field(raw_ce)
        mixed_output = parse_field(raw_mixed)

        prepared_data.append(
            {
                "ID": i,
                "title": title,
                "A": mixed_output if mixed else ce,
                "B": ce if mixed else mixed_output,
            }
        )
    return prepared_data


data = load_and_prep_data()


# --- 2. HELPERS FOR PROGRESS ---
def get_completed_ids():
    """Reads the results file to find which IDs are already done."""
    if os.path.exists(RESULTS_FILE):
        try:
            # Read CSV and get unique sample_ids
            df_res = pd.read_csv(RESULTS_FILE)
            if "sample_id" in df_res.columns:
                return set(df_res["sample_id"].unique())
        except Exception:
            return set()
    return set()


# Initialize session state for index
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0


def go_next():
    if st.session_state.current_idx < len(data) - 1:
        st.session_state.current_idx += 1


def go_prev():
    if st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1


# --- 3. SIDEBAR NAVIGATION ---
completed_ids = get_completed_ids()
total_samples = len(data)
completed_count = len(completed_ids)

with st.sidebar:
    st.header("📊 Progress")
    st.progress(completed_count / total_samples if total_samples > 0 else 0)
    st.write(f"**{completed_count} / {total_samples}** samples completed")

    st.markdown("---")
    st.subheader("Navigation")

    # Create a list of options for the dropdown with visual cues
    # Format: "ID: Title [✅]" or "ID: Title [ ]"
    options = [
        f"{d['ID']}: {d['title']} {'✅' if d['ID'] in completed_ids else ''}"
        for d in data
    ]

    # Find the index in the options list that matches the current session state ID
    current_option_index = st.session_state.current_idx

    selected_option = st.selectbox(
        "Jump to Sample:", options, index=current_option_index
    )

    # Update state if user used the dropdown
    selected_idx = int(selected_option.split(":")[0])
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
        st.rerun()

    # Button to find next pending
    if st.button("⏭️ Find Next Pending"):
        for i in range(len(data)):
            if data[i]["ID"] not in completed_ids:
                st.session_state.current_idx = i
                st.rerun()
        st.success("All samples completed!")

# --- 4. MAIN UI ---
if not data:
    st.error("No data found.")
    st.stop()

current_pair = data[st.session_state.current_idx]
sample_id = current_pair["ID"]

# Check status of THIS sample
is_already_done = sample_id in completed_ids

st.title("👨‍🍳 Recipe Evaluation")

# STATUS BANNER
if is_already_done:
    st.warning(
        f"⚠️ **Sample #{sample_id} is already evaluated.** Resubmitting will overwrite or duplicate the entry."
    )
else:
    st.info(f"📝 **Sample #{sample_id} - Pending Evaluation**")

st.markdown(f"## 🍽️ Recipe Request: *{current_pair['title']}*")
st.markdown("---")


# Render Recipe Function
def render_recipe(recipe_data):
    st.markdown("### 🥦 Ingredients")
    ingredients = recipe_data.get("ingredients", [])
    if isinstance(ingredients, list):
        for ing in ingredients:
            st.markdown(f"- {ing}")
    else:
        st.write(ingredients)

    st.markdown("### 🍳 Instructions")
    instructions = recipe_data.get("instructions", [])
    if isinstance(instructions, list):
        for i, step in enumerate(instructions, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write(instructions)


col1, col2 = st.columns(2)
with col1:
    st.header("Version A", divider="blue")
    render_recipe(current_pair["A"])
with col2:
    st.header("Version B", divider="red")
    render_recipe(current_pair["B"])

st.divider()

# --- 5. EVALUATION FORM ---
# CRITICAL FIX: The key includes sample_id (f"key_{sample_id}").
# When sample_id changes, Streamlit creates NEW widgets, resetting the values.

with st.form(key=f"form_{sample_id}"):  # The form key also changes

    st.markdown("#### 1. Comparison (A vs B)")
    c1, c2 = st.columns(2)
    with c1:
        p_ing = st.radio(
            "**Ingredients:** More appropriate?",
            ["A", "B", "Tie"],
            horizontal=True,
            index=None,
            key=f"p_ing_{sample_id}",
        )
        p_num = st.radio(
            "**Numbers:** More plausible?",
            ["A", "B", "Tie"],
            horizontal=True,
            index=None,
            key=f"p_num_{sample_id}",
        )
    with c2:
        p_proc = st.radio(
            "**Procedure:** Easier to follow?",
            ["A", "B", "Tie"],
            horizontal=True,
            index=None,
            key=f"p_proc_{sample_id}",
        )
        p_all = st.radio(
            "🏆 **OVERALL Preference:**",
            ["A", "B", "Tie"],
            horizontal=True,
            index=None,
            key=f"p_all_{sample_id}",
        )

    st.markdown("---")
    st.markdown("#### 2. Expert Check (Individual)")

    ce1, ce2 = st.columns(2)
    error_opts = [
        "Missing Core Ingredient",
        "Hallucination/Bizarre",
        "Bad Quantities",
        "Bad Times",
        "Bad Temperatures",
        "Step Mismatch",
        "Safety Issue",
        "Format Error",
    ]

    with ce1:
        st.markdown("**Version A Analysis**")
        a_cook = st.selectbox(
            "A: Cookable?",
            ["Yes", "Maybe (needs fix)", "No"],
            index=0,
            key=f"a_cook_{sample_id}",
        )
        a_trust = st.slider("A: Trust Score (1-5)", 1, 5, 3, key=f"a_trust_{sample_id}")
        a_err = st.multiselect("A: Severe Errors", error_opts, key=f"a_err_{sample_id}")

    with ce2:
        st.markdown("**Version B Analysis**")
        b_cook = st.selectbox(
            "B: Cookable?",
            ["Yes", "Maybe (needs fix)", "No"],
            index=0,
            key=f"b_cook_{sample_id}",
        )
        b_trust = st.slider("B: Trust Score (1-5)", 1, 5, 3, key=f"b_trust_{sample_id}")
        b_err = st.multiselect("B: Severe Errors", error_opts, key=f"b_err_{sample_id}")

    st.markdown("---")
    notes = st.text_area("Optional Notes", key=f"notes_{sample_id}")

    # Save Button
    submit = st.form_submit_button("💾 Save Evaluation", type="primary")

    if submit:
        # Validation
        if not all([p_ing, p_num, p_proc, p_all]):
            st.error("⚠️ Please fill in all comparison fields in Section 1.")
        else:
            new_row = {
                "sample_id": sample_id,
                "recipe_title": current_pair["title"],
                "pref_ingredients": p_ing,
                "pref_numbers": p_num,
                "pref_procedure": p_proc,
                "pref_overall": p_all,
                "A_cookable": a_cook,
                "A_trust": a_trust,
                "A_errors": ";".join(a_err) if a_err else "None",
                "B_cookable": b_cook,
                "B_trust": b_trust,
                "B_errors": ";".join(b_err) if b_err else "None",
                "notes": notes,
                "timestamp": pd.Timestamp.now(),
            }

            # Save to CSV
            df_entry = pd.DataFrame([new_row])
            write_header = not os.path.exists(RESULTS_FILE)
            df_entry.to_csv(RESULTS_FILE, mode="a", header=write_header, index=False)

            st.success("Saved! Moving to next...")

            # Logic to move to next
            go_next()
            st.rerun()

# Navigation Buttons at bottom
c_prev, c_next = st.columns([1, 1])
with c_prev:
    if st.button("⬅️ Previous", use_container_width=True):
        go_prev()
        st.rerun()
with c_next:
    if st.button("Skip / Next ➡️", use_container_width=True):
        go_next()
        st.rerun()
