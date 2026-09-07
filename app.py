import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

st.set_page_config(page_title="OIL Safety AI", page_icon="🦺", layout="wide")

MODEL_PATH = Path("sif_model.joblib")
DATA_PATH = Path("data/reports.csv")

RULES = {
    "Energy Isolation": ["isolation", "isolated", "lockout", "lock out", "energy"],
    "Hot Work": ["hot work", "welding", "cutting", "fire watch", "permit"],
    "Confined Space": ["confined space", "tank", "gas testing", "gas test", "standby person"],
    "Line of Fire": ["line of fire", "moving equipment", "vehicle", "revers", "struck by"],
    "Work at Height": ["height", "harness", "scaffold", "ladder", "edge protection"],
    "PPE": ["helmet", "gloves", "safety shoes", "ppe", "protective equipment"],
    "Gas/Chemical Exposure": ["gas leak", "gas smell", "chemical", "toxic", "leakage"],
    "Safety Device Bypass": ["bypass", "disabled safety", "guard was removed", "machine guard"],
}

RISK_WORDS = {
    "bypass": 25, "without harness": 20, "confined space": 20,
    "without gas": 20, "gas leak": 20, "line of fire": 20,
    "no barricading": 10, "slippery": 10, "without": 5,
    "removed": 10, "before energy isolation": 20
}

def extract_rules(text):
    t = text.lower()
    found = []
    for rule, keywords in RULES.items():
        if any(k in t for k in keywords):
            found.append(rule)
    return found

def risk_score(text, model_probability):
    t = text.lower()
    score = model_probability * 70
    for word, points in RISK_WORDS.items():
        if word in t:
            score += points
    return int(max(0, min(99, round(score))))

st.title("🦺 OIL Safety AI — SIF Precursor Detection")
st.caption("Prototype for SIH 2026 PS 26165")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

tab1, tab2 = st.tabs(["🔎 Analyze Report", "📊 Dashboard"])

with tab1:
    st.subheader("Enter an OIL safety report")
    text = st.text_area(
        "Report text",
        "Worker climbed on a pipeline without a harness. The surface was slippery and no barricading was present.",
        height=160
    )

    if st.button("Analyze Report", type="primary"):
        prob = float(model.predict_proba([text])[0][1])
        prediction = "SIF-Potential" if prob >= 0.5 else "Non-SIF"
        score = risk_score(text, prob)
        rules = extract_rules(text)

        c1, c2, c3 = st.columns(3)
        c1.metric("SIF Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Score", f"{score}/100")
        c3.metric("Classification", prediction)

        if score >= 70:
            st.error("🔴 HIGH PRIORITY — Immediate safety review recommended.")
        elif score >= 40:
            st.warning("🟠 MEDIUM PRIORITY — Safety review recommended.")
        else:
            st.success("🟢 LOW PRIORITY — No strong SIF precursor detected.")

        st.subheader("Relevant Life-Saving Rules / Risk Tags")
        if rules:
            for rule in rules:
                st.write("•", rule)
        else:
            st.write("No rule confidently matched in this prototype.")

        st.subheader("Detected risk indicators")
        indicators = []
        t = text.lower()
        for word in RISK_WORDS:
            if word in t:
                indicators.append(word)
        st.write(", ".join(indicators) if indicators else "No keyword indicator detected.")

        st.info("Prototype note: the score is a prioritization signal, not a prediction of an actual accident or death.")

with tab2:
    st.subheader("Report Dashboard")
    df = pd.read_csv(DATA_PATH)
    probs = model.predict_proba(df["text"])[:, 1]
    df["sif_probability"] = probs
    df["risk_score"] = [risk_score(t, p) for t, p in zip(df["text"], probs)]
    df["classification"] = df["sif_probability"].apply(lambda x: "SIF-Potential" if x >= 0.5 else "Non-SIF")

    a,b,c,d = st.columns(4)
    a.metric("Total Reports", len(df))
    b.metric("SIF-Potential", int((df["classification"]=="SIF-Potential").sum()))
    c.metric("High Priority", int((df["risk_score"]>=70).sum()))
    d.metric("Average Risk", f"{df['risk_score'].mean():.1f}/100")

    st.subheader("Highest-priority reports")
    st.dataframe(
        df.sort_values("risk_score", ascending=False)[
            ["report_id","text","classification","risk_score"]
        ].head(10),
        use_container_width=True,
        hide_index=True
    )

    rule_counts = {}
    for text in df["text"]:
        for rule in extract_rules(text):
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    if rule_counts:
        st.subheader("Recurring risk patterns")
        rc = pd.DataFrame(
            sorted(rule_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["Risk / Life-Saving Rule","Reports"]
        )
        st.bar_chart(rc.set_index("Risk / Life-Saving Rule"))
