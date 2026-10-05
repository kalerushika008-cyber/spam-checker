import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spam Checker", page_icon="📩", layout="wide")

LABELS = ["spam", "not spam"]
DATA_FILE = "training_data.csv"

DEFAULT = pd.DataFrame({
    "message": [
        "Win a free iPhone now", "Claim your lottery prize", "You won 1 crore click here",
        "Free recharge offer limited time", "Get cash instantly, no documents",
        "Are you coming to class today", "Submit the assignment by Friday",
        "Let's meet for lunch", "Mom called, please call back", "Project meeting at 5 pm",
    ],
    "label": ["spam"] * 5 + ["not spam"] * 5,
})

EXAMPLES = [
    "",
    "Win free cash now",
    "Claim your free recharge offer",
    "Meeting at 5 pm tomorrow",
    "Please submit the project by Friday",
]

# ---------- Saving and loading ----------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return DEFAULT.copy()

def save_data():
    st.session_state.data.to_csv(DATA_FILE, index=False)

if "data" not in st.session_state:
    st.session_state.data = load_data()
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- The AI (Naive Bayes with numpy) ----------
def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())

def train(df):
    vocab = sorted({w for m in df["message"] for w in tokenize(m)})
    index = {w: i for i, w in enumerate(vocab)}
    word_logp, priors = {}, {}
    for label in LABELS:
        rows = df[df["label"] == label]
        counts = np.ones(len(vocab))
        for m in rows["message"]:
            for w in tokenize(m):
                counts[index[w]] += 1
        word_logp[label] = np.log(counts / counts.sum())
        priors[label] = np.log((len(rows) + 1) / (len(df) + 2))
    return index, word_logp, priors

def predict(text):
    known = [w for w in tokenize(text) if w in index]
    if not known:
        return None
    scores = np.array([
        priors[l] + sum(word_logp[l][index[w]] for w in known) for l in LABELS
    ])
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    return dict(zip(LABELS, probs)), known

def risk_level(spam_prob):
    if spam_prob >= 0.7:
        return "🔴 High"
    if spam_prob >= 0.3:
        return "🟡 Medium"
    return "🟢 Low"

# ---------- Sidebar: teach the AI ----------
with st.sidebar:
    st.header("🧠 Teach the AI")

    with st.form("add_form", clear_on_submit=True):
        new_msg = st.text_area("New example message")
        new_label = st.radio("This message is:", LABELS, horizontal=True)
        add = st.form_submit_button("➕ Add example")
    if add:
        if new_msg.strip():
            row = pd.DataFrame({"message": [new_msg.strip()], "label": [new_label]})
            st.session_state.data = pd.concat(
                [st.session_state.data, row], ignore_index=True
            )
            save_data()
            st.success("Added! The AI has retrained.")
        else:
            st.warning("Type a message first.")

    st.divider()
    up = st.file_uploader("Upload CSV (columns: message, label)", type="csv")
    if up is not None and st.button("📥 Load CSV into training data"):
        try:
            new = pd.read_csv(up)
            new.columns = [c.lower().strip() for c in new.columns]
            new = new[["message", "label"]].dropna()
            new["label"] = new["label"].astype(str).str.lower().str.strip()
            new = new[new["label"].isin(LABELS)]
            st.session_state.data = (
                pd.concat([st.session_state.data, new])
                .drop_duplicates()
                .reset_index(drop=True)
            )
            save_data()
            st.success(f"Loaded {len(new)} rows!")
        except Exception:
            st.error("CSV needs 'message' and 'label' columns.")

    st.divider()
    if st.button("♻️ Reset to original data"):
        st.session_state.data = DEFAULT.copy()
        st.session_state.history = []
        save_data()
        st.rerun()

# ---------- Train on the current data ----------
data = st.session_state.data
if data.empty:
    st.warning("No training data. Click 'Reset to original data' in the sidebar.")
    st.stop()

index, word_logp, priors = train(data)

# ---------- Main page ----------
st.title("📩 Spam Message Checker")
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Check", "📦 Bulk check", "📊 History & stats", "🧾 Training data"]
)

# --- Tab 1: single check ---
with tab1:
    choice = st.selectbox("Pick an example (optional):", EXAMPLES)
    msg = st.text_area("Paste a message:", value=choice, height=120)

    if st.button("🔍 Check message", type="primary"):
        result = predict(msg)
        if result is None:
            st.warning("I don't know any of these words yet. Teach me in the sidebar!")
        else:
            probs, known = result
            label = max(probs, key=probs.get)
            conf = probs[label] * 100

            if label == "spam":
                st.error(f"🚨 SPAM ({conf:.1f}% sure)")
            else:
                st.success(f"✅ NOT SPAM ({conf:.1f}% sure)")

            st.progress(float(probs["spam"]), text=f"Spam score: {probs['spam']*100:.0f}%")
            st.write(f"**Risk level:** {risk_level(probs['spam'])}")

            # Highlight words: red = spammy, green = safe
            parts = []
            for w in tokenize(msg):
                if w in index:
                    ratio = word_logp["spam"][index[w]] - word_logp["not spam"][index[w]]
                    parts.append(f":red[{w}]" if ratio > 0 else f":green[{w}]")
                else:
                    parts.append(w)
            st.write("**Word view:** " + " ".join(parts))

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": msg,
                "result": label,
                "confidence %": round(conf, 1),
            })

# --- Tab 2: bulk check ---
with tab2:
    st.subheader("Check many messages at once")
    bulk = st.text_area("One message per line:", height=150)
    file = st.file_uploader("...or upload a CSV with a 'message' column",
                            type="csv", key="bulk_file")

    msgs = [l.strip() for l in bulk.splitlines() if l.strip()]
    if file is not None:
        df_in = pd.read_csv(file)
        if "message" in df_in.columns:
            msgs += df_in["message"].dropna().astype(str).tolist()
        else:
            st.error("Your CSV needs a column named 'message'.")

    if st.button("Check all") and msgs:
        rows = []
        for m in msgs:
            r = predict(m)
            if r is None:
                rows.append({"message": m, "result": "unknown", "spam %": np.nan})
            else:
                p, _ = r
                rows.append({"message": m, "result": max(p, key=p.get),
                             "spam %": round(p["spam"] * 100, 1)})
        st.session_state.bulk_out = pd.DataFrame(rows)

    if "bulk_out" in st.session_state:
        out = st.session_state.bulk_out
        st.dataframe(out)
        st.download_button("⬇️ Download results", out.to_csv(index=False),
                           "results.csv", "text/csv")

# --- Tab 3: history ---
with tab3:
    hist = pd.DataFrame(st.session_state.history)
    if hist.empty:
        st.info("No checks yet. Go to the Check tab and try a message.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total checks", len(hist))
        c2.metric("Spam found", int((hist["result"] == "spam").sum()))
        c3.metric("Not spam", int((hist["result"] == "not spam").sum()))
        st.bar_chart(hist["result"].value_counts())
        st.dataframe(hist)
        st.download_button("⬇️ Download history", hist.to_csv(index=False),
                           "history.csv", "text/csv")
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# --- Tab 4: training data ---
with tab4:
    c1, c2, c3 = st.columns(3)
    c1.metric("Training messages", len(data))
    c2.metric("Spam examples", int((data["label"] == "spam").sum()))
    c3.metric("Not spam examples", int((data["label"] == "not spam").sum()))

    st.subheader("Top spam words the AI learned")
    ratio = {w: word_logp["spam"][i] - word_logp["not spam"][i] for w, i in index.items()}
    st.bar_chart(pd.Series(ratio).sort_values(ascending=False).head(10))

    st.subheader("All training data")
    st.dataframe(data)
    st.download_button("⬇️ Download training data", data.to_csv(index=False),
                       "training_data.csv", "text/csv")