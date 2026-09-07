import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

st.set_page_config(page_title="OIL Safety AI", page_icon="🦺", layout="wide")

MODEL_PATH = Path("sif_model.joblib")
DATA_PATH = Path("data/reports_v2.csv")

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

    SAFETY_ACTIONS = {
    "Energy Isolation": [
        "Confirm complete energy isolation before starting work.",
        "Verify lockout/tagout (LOTO) controls.",
        "Do not begin maintenance until isolation is confirmed."
    ],
    "Hot Work": [
        "Verify the hot-work permit.",
        "Ensure a fire watch is assigned.",
        "Check the area for flammable materials before starting work."
    ],
    "Confined Space": [
        "Perform gas testing before entry.",
        "Assign a trained standby person.",
        "Verify the confined-space permit.",
        "Confirm emergency rescue arrangements.",
        "Stop work until required controls are verified."
    ],
    "Line of Fire": [
        "Keep workers away from the line of fire.",
        "Establish a safe exclusion zone around moving equipment.",
        "Use a spotter where required."
    ],
    "Work at Height": [
        "Use an approved fall-protection system.",
        "Verify harness and anchorage points.",
        "Ensure proper barricading and edge protection."
    ],
    "PPE": [
        "Verify that required PPE is being worn.",
        "Replace damaged or missing PPE before starting work."
    ],
    "Gas/Chemical Exposure": [
        "Stop work and isolate the affected area.",
        "Perform appropriate gas or chemical monitoring.",
        "Use the required protective equipment."
    ],
    "Safety Device Bypass": [
        "Stop the operation.",
        "Restore the required safety device or guard.",
        "Do not operate equipment with safety controls bypassed."
    ]
}

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

        st.subheader("🚨 Recommended Safety Actions")

        if rules:
            for rule in rules:
                if rule in SAFETY_ACTIONS:
                    st.write(f"**{rule}**")
                    for action in SAFETY_ACTIONS[rule]:
                        st.write("•", action)
        else:
            st.write("No specific safety action identified.")

        st.info("Prototype note: the score is a prioritization signal, not a prediction of an actual accident or death.")

with tab2:
    st.subheader("📊 OIL Safety Risk Dashboard")

    # Load the complete safety-report dataset
    df = pd.read_csv(DATA_PATH)

    # AI prediction
    probs = model.predict_proba(df["text"])[:, 1]

    df["sif_probability"] = probs
    df["risk_score"] = [
        risk_score(t, p) for t, p in zip(df["text"], probs)
    ]

    df["classification"] = df["sif_probability"].apply(
        lambda x: "SIF-Potential" if x >= 0.5 else "Non-SIF"
    )

    # ---------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------

    st.subheader("🔎 Filter Reports")

    f1, f2, f3 = st.columns(3)

    with f1:
        site_filter = st.selectbox(
            "Site",
            ["All"] + sorted(df["site"].dropna().unique().tolist())
        )

    with f2:
        department_filter = st.selectbox(
            "Department",
            ["All"] + sorted(df["department"].dropna().unique().tolist())
        )

    with f3:
        risk_filter = st.selectbox(
            "Classification",
            ["All", "SIF-Potential", "Non-SIF"]
        )

    filtered_df = df.copy()

    if site_filter != "All":
        filtered_df = filtered_df[
            filtered_df["site"] == site_filter
        ]

    if department_filter != "All":
        filtered_df = filtered_df[
            filtered_df["department"] == department_filter
        ]

    if risk_filter != "All":
        filtered_df = filtered_df[
            filtered_df["classification"] == risk_filter
        ]

    # ---------------------------------------------------------
    # KEY METRICS
    # ---------------------------------------------------------

    st.subheader("📌 Risk Summary")

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Reports",
        len(filtered_df)
    )

    b.metric(
        "SIF-Potential",
        int(
            (filtered_df["classification"] == "SIF-Potential").sum()
        )
    )

    c.metric(
        "High Priority",
        int(
            (filtered_df["risk_score"] >= 70).sum()
        )
    )

    d.metric(
        "Average Risk",
        f"{filtered_df['risk_score'].mean():.1f}/100"
        if len(filtered_df) > 0
        else "0/100"
    )

    # ---------------------------------------------------------
    # HIGH PRIORITY REPORTS
    # ---------------------------------------------------------

    st.subheader("🚨 Highest-Priority Reports")

    if len(filtered_df) > 0:

        high_risk = filtered_df.sort_values(
            "risk_score",
            ascending=False
        ).head(10)

        st.dataframe(
            high_risk[
                [
                    "report_id",
                    "text",
                    "classification",
                    "risk_score",
                    "site",
                    "department",
                    "activity"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No reports match the selected filters.")

    # ---------------------------------------------------------
    # SITE ANALYSIS
    # ---------------------------------------------------------

    st.subheader("🏭 Risk by Site")

    site_analysis = (
        filtered_df
        .groupby("site")
        .agg(
            Reports=("report_id", "count"),
            SIF_Potential=("classification",
                           lambda x: (x == "SIF-Potential").sum()),
            Average_Risk=("risk_score", lambda x: round(x.mean(), 1))
        )
        .reset_index()
        .sort_values("Average_Risk", ascending=False)
    )

    if not site_analysis.empty:
        st.dataframe(
            site_analysis,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            site_analysis.set_index("site")["Average_Risk"]
        )

    # ---------------------------------------------------------
    # DEPARTMENT ANALYSIS
    # ---------------------------------------------------------

    st.subheader("🏢 Risk by Department")

    department_analysis = (
        filtered_df
        .groupby("department")
        .agg(
            Reports=("report_id", "count"),
            SIF_Potential=("classification",
                           lambda x: (x == "SIF-Potential").sum()),
            Average_Risk=("risk_score", lambda x: round(x.mean(), 1))
        )
        .reset_index()
        .sort_values("Average_Risk", ascending=False)
    )

    if not department_analysis.empty:
        st.dataframe(
            department_analysis,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            department_analysis.set_index("department")["Average_Risk"]
        )

    # ---------------------------------------------------------
    # ACTIVITY ANALYSIS
    # ---------------------------------------------------------

    st.subheader("🔧 Risk by Activity")

    activity_analysis = (
        filtered_df
        .groupby("activity")
        .agg(
            Reports=("report_id", "count"),
            SIF_Potential=("classification",
                           lambda x: (x == "SIF-Potential").sum()),
            Average_Risk=("risk_score", lambda x: round(x.mean(), 1))
        )
        .reset_index()
        .sort_values("Average_Risk", ascending=False)
    )

    if not activity_analysis.empty:
        st.dataframe(
            activity_analysis,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            activity_analysis.set_index("activity")["Average_Risk"]
        )

    # ---------------------------------------------------------
    # RISK CATEGORY
    # ---------------------------------------------------------

    st.subheader("⚠️ Risk Categories")

    risk_category_analysis = (
        filtered_df
        .groupby("risk_category")
        .size()
        .reset_index(name="Reports")
        .sort_values("Reports", ascending=False)
    )

    if not risk_category_analysis.empty:
        st.bar_chart(
            risk_category_analysis.set_index("risk_category")
        )

    # ---------------------------------------------------------
    # LIFE-SAVING RULES
    # ---------------------------------------------------------

    st.subheader("🛡️ Recurring Life-Saving Rules")

    rule_counts = {}

    for text in filtered_df["text"]:

        for rule in extract_rules(text):

            rule_counts[rule] = rule_counts.get(rule, 0) + 1

    if rule_counts:

        rc = pd.DataFrame(
            sorted(
                rule_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            columns=[
                "Life-Saving Rule",
                "Reports"
            ]
        )

        st.bar_chart(
            rc.set_index("Life-Saving Rule")
        )

    else:

        st.info(
            "No recurring Life-Saving Rules detected."
        )
