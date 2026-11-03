
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# KIDNEY DIGITAL TWIN — HEOR MODULE
# ============================================================

st.set_page_config(
    page_title="Kidney Digital Twin | HEOR",
    page_icon="🩺",
    layout="wide",
)

st.title("Kidney Digital Twin")
st.subheader("Health Economics and Outcomes Research")

FILE_NAME = "Digital_Twin_Kidney_Synthetic_Shared_Dataset (2).xlsx"
FILE_PATH = Path(__file__).resolve().parent / FILE_NAME

RESOURCE_COLUMNS = [
    "Annual_Lab_Visits",
    "Annual_Outpatient_Visits",
    "Annual_Hospitalizations",
    "Medication_Resource_Units",
    "Illustrative_Annual_Cost",
]

FEATURE_COLUMNS = [
    "Age", "Sex", "Diabetes", "Hypertension", "BMI", "eGFR",
    "Creatinine", "UACR", "SBP", "DBP", "HbA1c", "Potassium",
    "Hemoglobin",
]

CATEGORICAL_COLUMNS = ["Sex", "Diabetes", "Hypertension"]
NUMERIC_COLUMNS = [
    c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS
]

STATE_ORDER = [
    "Stable",
    "Moderate Progression",
    "Advanced Progression",
]

# ============================================================
# COST ASSUMPTIONS — SYNTHETIC DATASET
# ============================================================

COST_ASSUMPTIONS = {
    "Stable": {
        "Base Cost": 1000,
        "Lab Visit Cost": 120,
        "Outpatient Visit Cost": 180,
        "Hospitalization Cost": 1500,
        "Medication Unit Cost": 220,
    },
    "Moderate Progression": {
        "Base Cost": 1800,
        "Lab Visit Cost": 140,
        "Outpatient Visit Cost": 220,
        "Hospitalization Cost": 1800,
        "Medication Unit Cost": 260,
    },
    "Advanced Progression": {
        "Base Cost": 3500,
        "Lab Visit Cost": 180,
        "Outpatient Visit Cost": 260,
        "Hospitalization Cost": 2400,
        "Medication Unit Cost": 320,
    },
}

# ============================================================
# LOAD EXCEL WORKBOOK
# ============================================================

if not FILE_PATH.exists():
    st.error("Excel workbook not found.")
    st.write("Expected file path:")
    st.code(str(FILE_PATH))
    st.info("Keep the Excel workbook in the same folder as app.py.")
    st.stop()


@st.cache_data
def load_workbook(path_string):
    path = Path(path_string)

    patient = pd.read_excel(path, sheet_name="Patient_Master")
    longitudinal = pd.read_excel(
        path, sheet_name="Longitudinal_Data"
    )
    prediction = pd.read_excel(
        path, sheet_name="Prediction_Targets"
    )
    resource = pd.read_excel(
        path, sheet_name="Resource_Economic"
    )

    longitudinal["Date"] = pd.to_datetime(
        longitudinal["Date"], errors="coerce"
    )

    latest = (
        longitudinal.sort_values("Date")
        .dropna(subset=["Date"])
        .groupby("Patient_ID", as_index=False)
        .tail(1)
    )

    training = (
        patient.merge(latest, on="Patient_ID", how="left")
        .merge(
            prediction[
                [
                    "Patient_ID",
                    "Future_eGFR_6m",
                    "Future_eGFR_12m",
                    "Projected_State_12m",
                    "Synthetic_Risk_Band",
                ]
            ],
            on="Patient_ID",
            how="left",
        )
    )

    patient_data = prediction.merge(
        resource,
        on=["Patient_ID", "Projected_State_12m"],
        how="left",
        validate="one_to_one",
    )

    return (
        patient,
        longitudinal,
        prediction,
        resource,
        training,
        patient_data,
    )


try:
    (
        patient_master,
        longitudinal,
        prediction,
        resource,
        training,
        patient_data,
    ) = load_workbook(str(FILE_PATH))

except Exception as exc:
    st.error(
        "The workbook could not be loaded. Check its filename, "
        "sheet names, and columns."
    )
    st.exception(exc)
    st.stop()


missing_columns = [
    c for c in RESOURCE_COLUMNS if c not in resource.columns
]

if missing_columns:
    st.error(
        "Missing Resource_Economic columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


resource_summary = (
    patient_data.groupby("Projected_State_12m", dropna=False)[
        RESOURCE_COLUMNS
    ]
    .mean()
    .reset_index()
)

resource_summary["Projected_State_12m"] = pd.Categorical(
    resource_summary["Projected_State_12m"],
    categories=STATE_ORDER,
    ordered=True,
)

resource_summary = resource_summary.sort_values(
    "Projected_State_12m"
)

PATIENT_IDS = sorted(
    patient_data["Patient_ID"].dropna().astype(str).unique()
)


# ============================================================
# MODEL PREPROCESSOR
# ============================================================

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
                ]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Patient Profile",
        "Resource Utilization",
        "Economic Consequence",
        "Burden of Illness & Reimbursement",
        "Predict a Hypothetical Patient",
        "ISA / ISR",
        "Cost Assumptions",
    ],
)

st.sidebar.metric(
    "Patients",
    int(patient_master["Patient_ID"].nunique()),
)


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Patients",
        int(patient_master["Patient_ID"].nunique()),
    )
    c2.metric("Longitudinal Records", int(len(longitudinal)))
    c3.metric("Resource Records", int(len(resource)))

    st.subheader("Projected-State Summary")

    overview = resource_summary.rename(
        columns={
            "Projected_State_12m": "Projected State",
            "Annual_Lab_Visits": "Average Lab Visits / Year",
            "Annual_Outpatient_Visits": "Average Outpatient Visits / Year",
            "Annual_Hospitalizations": "Average Hospitalizations / Year",
            "Medication_Resource_Units": "Average Medication Units / Year",
            "Illustrative_Annual_Cost": "Average Annual Cost (Cost Units)",
        }
    )

    st.dataframe(
        overview.round(2),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        resource_summary,
        x="Projected_State_12m",
        y="Illustrative_Annual_Cost",
        color="Projected_State_12m",
        title="Average Annual Cost by Projected State",
        labels={
            "Projected_State_12m": "Projected State",
            "Illustrative_Annual_Cost": "Average Annual Cost (Cost Units)",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. PATIENT PROFILE
# ============================================================

elif page == "Patient Profile":

    st.header("Patient Profile")

    selected_id = st.selectbox(
        "Select Patient ID",
        PATIENT_IDS,
    )

    row = patient_data[
        patient_data["Patient_ID"].astype(str) == selected_id
    ].iloc[0]

    st.subheader("Prediction")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current eGFR",
        "—" if pd.isna(row["Last_eGFR"])
        else f"{row['Last_eGFR']:.1f}",
    )

    c2.metric(
        "6-Month eGFR",
        "—" if pd.isna(row["Future_eGFR_6m"])
        else f"{row['Future_eGFR_6m']:.1f}",
    )

    c3.metric(
        "12-Month eGFR",
        "—" if pd.isna(row["Future_eGFR_12m"])
        else f"{row['Future_eGFR_12m']:.1f}",
    )

    c4.metric(
        "Risk Band",
        str(row.get("Synthetic_Risk_Band", "—")),
    )

    st.write(
        "**Projected 12-month state:**",
        row["Projected_State_12m"],
    )

    st.write(
        "**Deterioration flag:**",
        row.get("Synthetic_Deterioration_Flag", "—"),
    )

    st.subheader("Annual Resource Utilization")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Lab Visits", f"{row['Annual_Lab_Visits']:.1f}")
    c2.metric(
        "Outpatient Visits",
        f"{row['Annual_Outpatient_Visits']:.1f}",
    )
    c3.metric(
        "Hospitalizations",
        f"{row['Annual_Hospitalizations']:.1f}",
    )
    c4.metric(
        "Medication Units",
        f"{row['Medication_Resource_Units']:.1f}",
    )

    st.subheader("Economic Consequence")

    st.metric(
        "Illustrative Annual Cost",
        f"{row['Illustrative_Annual_Cost']:,.0f} "
        f"{row.get('Cost_Unit', 'Cost Units')}",
    )


# ============================================================
# 3. RESOURCE UTILIZATION
# ============================================================

elif page == "Resource Utilization":

    st.header("Resource Utilization by Projected State")

    display = resource_summary.rename(
        columns={
            "Projected_State_12m": "Projected State",
            "Annual_Lab_Visits": "Average Lab Visits / Year",
            "Annual_Outpatient_Visits": "Average Outpatient Visits / Year",
            "Annual_Hospitalizations": "Average Hospitalizations / Year",
            "Medication_Resource_Units": "Average Medication Units / Year",
            "Illustrative_Annual_Cost": "Average Annual Cost (Cost Units)",
        }
    )

    st.dataframe(
        display.round(2),
        use_container_width=True,
        hide_index=True,
    )

    charts = [
        (
            "Annual_Lab_Visits",
            "Average Annual Laboratory Visits",
            "Visits / Year",
        ),
        (
            "Annual_Outpatient_Visits",
            "Average Annual Outpatient Visits",
            "Visits / Year",
        ),
        (
            "Annual_Hospitalizations",
            "Average Annual Hospitalizations",
            "Hospitalizations / Year",
        ),
        (
            "Medication_Resource_Units",
            "Average Medication Resource Units",
            "Units / Year",
        ),
    ]

    for col, title, label in charts:

        fig = px.bar(
            resource_summary,
            x="Projected_State_12m",
            y=col,
            color="Projected_State_12m",
            title=title,
            labels={
                "Projected_State_12m": "Projected State",
                col: label,
            },
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4. ECONOMIC CONSEQUENCE
# ============================================================

elif page == "Economic Consequence":

    st.header("Economic Consequence")

    selected_id = st.selectbox(
        "Select Patient ID",
        PATIENT_IDS,
        key="economic_patient",
    )

    row = patient_data[
        patient_data["Patient_ID"].astype(str) == selected_id
    ].iloc[0]

    st.write(
        "**Projected State:**",
        row["Projected_State_12m"],
    )

    st.metric(
        "Workbook Annual Cost",
        f"{row['Illustrative_Annual_Cost']:,.0f} "
        f"{row.get('Cost_Unit', 'Cost Units')}",
    )

    components = pd.DataFrame({
        "Resource": [
            "Laboratory Visits",
            "Outpatient Visits",
            "Hospitalizations",
            "Medication Resource Units",
        ],
        "Annual Utilization": [
            row["Annual_Lab_Visits"],
            row["Annual_Outpatient_Visits"],
            row["Annual_Hospitalizations"],
            row["Medication_Resource_Units"],
        ],
    })

    st.subheader("Patient Resource-Use Profile")

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        components,
        x="Resource",
        y="Annual Utilization",
        title="Annual Resource Utilization",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5. BURDEN OF ILLNESS AND REIMBURSEMENT
# ============================================================

elif page == "Burden of Illness & Reimbursement":

    st.header("Burden of Illness and Reimbursement")

    total_cost = float(
        resource["Illustrative_Annual_Cost"].fillna(0).sum()
    )

    n_patients = int(resource["Patient_ID"].nunique())

    c1, c2 = st.columns(2)

    c1.metric("Patients", n_patients)
    c2.metric(
        "Total Annual Cost",
        f"{total_cost:,.0f} Cost Units",
    )

    current_rate = st.slider(
        "Current payer reimbursement (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    current_payer = total_cost * current_rate / 100

    scenarios = pd.DataFrame([
        {
            "Scenario": "Current Baseline",
            "Reimbursement (%)": current_rate,
            "Government / Payer Budget": current_payer,
            "Patient Out-of-Pocket": total_cost - current_payer,
            "Additional Payer Budget": 0.0,
        },
        {
            "Scenario": "100% Reimbursement",
            "Reimbursement (%)": 100,
            "Government / Payer Budget": total_cost,
            "Patient Out-of-Pocket": 0.0,
            "Additional Payer Budget": total_cost - current_payer,
        },
        {
            "Scenario": "50% Reimbursement",
            "Reimbursement (%)": 50,
            "Government / Payer Budget": total_cost * 0.50,
            "Patient Out-of-Pocket": total_cost * 0.50,
            "Additional Payer Budget": total_cost * 0.50 - current_payer,
        },
    ])

    st.dataframe(
        scenarios.round(2),
        use_container_width=True,
        hide_index=True,
    )

    chart_data = scenarios[
        scenarios["Scenario"] != "Current Baseline"
    ].melt(
        id_vars="Scenario",
        value_vars=[
            "Government / Payer Budget",
            "Patient Out-of-Pocket",
        ],
        var_name="Cost Bearer",
        value_name="Cost Units",
    )

    fig = px.bar(
        chart_data,
        x="Scenario",
        y="Cost Units",
        color="Cost Bearer",
        barmode="group",
        title="Payer Budget and Patient Share",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 6. PREDICT A HYPOTHETICAL PATIENT
# ============================================================

elif page == "Predict a Hypothetical Patient":

    st.header("Predict a Hypothetical Patient")

    missing_features = [
        c for c in FEATURE_COLUMNS if c not in training.columns
    ]

    if missing_features:
        st.error(
            "Missing model inputs: " + ", ".join(missing_features)
        )
        st.stop()

    model_data = training.dropna(
        subset=["Projected_State_12m"]
    ).copy()

    X = model_data[FEATURE_COLUMNS]
    y_state = model_data["Projected_State_12m"].astype(str)

    y_6m = pd.to_numeric(
        model_data["Future_eGFR_6m"], errors="coerce"
    )
    y_12m = pd.to_numeric(
        model_data["Future_eGFR_12m"], errors="coerce"
    )

    try:
        classifier = Pipeline([
            ("preprocessor", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ])

        classifier.fit(X, y_state)

        valid_6m = y_6m.notna()
        valid_12m = y_12m.notna()

        if not valid_6m.any() or not valid_12m.any():
            st.error("No valid future eGFR targets are available.")
            st.stop()

        regressor_6m = Pipeline([
            ("preprocessor", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                ),
            ),
        ])

        regressor_12m = Pipeline([
            ("preprocessor", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                ),
            ),
        ])

        regressor_6m.fit(
            X.loc[valid_6m],
            y_6m.loc[valid_6m],
        )

        regressor_12m.fit(
            X.loc[valid_12m],
            y_12m.loc[valid_12m],
        )

    except Exception as exc:
        st.error("The prediction models could not be trained.")
        st.exception(exc)
        st.stop()

    st.subheader("Patient Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", 18, 100, 55)
        sex = st.selectbox(
            "Sex", ["Female", "Male", "Other/Unknown"]
        )
        bmi = st.number_input(
            "BMI", 10.0, 70.0, 25.0, step=0.5
        )
        diabetes = st.selectbox(
            "Diabetes", ["No", "Yes", "Unknown"]
        )
        hypertension = st.selectbox(
            "Hypertension", ["No", "Yes", "Unknown"]
        )

    with c2:
        egfr = st.number_input(
            "Current eGFR", 1.0, 180.0, 60.0
        )
        creatinine = st.number_input(
            "Creatinine", 0.1, 20.0, 1.2, step=0.1
        )
        uacr = st.number_input(
            "UACR", 0.0, 10000.0, 50.0, step=5.0
        )
        sbp = st.number_input(
            "Systolic BP", 60, 260, 130
        )
        dbp = st.number_input(
            "Diastolic BP", 30, 160, 80
        )

    with c3:
        hba1c = st.number_input(
            "HbA1c", 3.0, 25.0, 6.5, step=0.1
        )
        potassium = st.number_input(
            "Potassium", 1.0, 10.0, 4.5, step=0.1
        )
        hemoglobin = st.number_input(
            "Hemoglobin", 3.0, 25.0, 12.0, step=0.1
        )

    if st.button("Generate Prediction", type="primary"):

        hypothetical = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "BMI": bmi,
            "eGFR": egfr,
            "Creatinine": creatinine,
            "UACR": uacr,
            "SBP": sbp,
            "DBP": dbp,
            "HbA1c": hba1c,
            "Potassium": potassium,
            "Hemoglobin": hemoglobin,
        }])

        predicted_state = str(
            classifier.predict(hypothetical)[0]
        )
        predicted_6m = float(
            regressor_6m.predict(hypothetical)[0]
        )
        predicted_12m = float(
            regressor_12m.predict(hypothetical)[0]
        )

        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted 6-Month eGFR",
            f"{predicted_6m:.1f}",
        )
        c2.metric(
            "Predicted 12-Month eGFR",
            f"{predicted_12m:.1f}",
        )
        c3.metric(
            "Projected 12-Month State",
            predicted_state,
        )

        scores = pd.DataFrame({
            "Projected State": classifier.classes_,
            "Model Score (%)": (
                classifier.predict_proba(hypothetical)[0] * 100
            ),
        }).sort_values(
            "Model Score (%)",
            ascending=False,
        )

        st.subheader("Model Scores")

        st.dataframe(
            scores.round(2),
            use_container_width=True,
            hide_index=True,
        )

        matched = resource_summary[
            resource_summary["Projected_State_12m"].astype(str)
            == predicted_state
        ]

        if matched.empty:
            st.warning(
                "No resource-use data is available for the predicted state."
            )
        else:
            r = matched.iloc[0]

            st.subheader("Linked Resource and Economic Outcome")

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Lab Visits / Year",
                f"{r['Annual_Lab_Visits']:.2f}",
            )
            c2.metric(
                "Outpatient Visits / Year",
                f"{r['Annual_Outpatient_Visits']:.2f}",
            )
            c3.metric(
                "Hospitalizations / Year",
                f"{r['Annual_Hospitalizations']:.2f}",
            )
            c4.metric(
                "Medication Units / Year",
                f"{r['Medication_Resource_Units']:.2f}",
            )
            c5.metric(
                "Average Annual Cost",
                f"{r['Illustrative_Annual_Cost']:,.0f} Cost Units",
            )


# ============================================================
# 7. ISA / ISR
# ============================================================

elif page == "ISA / ISR":

    st.header("ISA / ISR")

    st.subheader("Patient-Level Inputs")

    isa_data = patient_data[
        [
            "Patient_ID",
            "Projected_State_12m",
            "Last_eGFR",
            "Future_eGFR_12m",
            "Illustrative_Annual_Cost",
            "Cost_Unit",
        ]
    ].rename(
        columns={
            "Patient_ID": "Patient ID",
            "Projected_State_12m": "Projected State",
            "Last_eGFR": "Current eGFR",
            "Future_eGFR_12m": "12-Month eGFR",
            "Illustrative_Annual_Cost": "Annual Cost",
            "Cost_Unit": "Cost Unit",
        }
    )

    st.dataframe(
        isa_data,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Calculation Details")

    st.dataframe(
        pd.DataFrame([
            {
                "Item": "ISA/ISR definition",
                "Status": "Awaiting confirmed definition",
            },
            {
                "Item": "Formula",
                "Status": "Awaiting confirmed formula",
            },
            {
                "Item": "Calculated value",
                "Status": "Not calculated",
            },
        ]),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 8. COST ASSUMPTIONS
# ============================================================

elif page == "Cost Assumptions":

    st.header("Cost Assumptions")

    st.subheader("State-Specific Cost Coefficients")

    assumptions = pd.DataFrame([
        {"Projected State": state, **values}
        for state, values in COST_ASSUMPTIONS.items()
    ])

    st.dataframe(
        assumptions,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Annual Cost Formula")

    st.markdown("**Stable**")
    st.latex(
        r"C = 1000 + 120L + 180O + 1500H + 220M"
    )

    st.markdown("**Moderate Progression**")
    st.latex(
        r"C = 1800 + 140L + 220O + 1800H + 260M"
    )

    st.markdown("**Advanced Progression**")
    st.latex(
        r"C = 3500 + 180L + 260O + 2400H + 320M"
    )

    st.markdown(
        "**C** = annual cost; **L** = annual laboratory visits; "
        "**O** = annual outpatient visits; **H** = annual hospitalizations; "
        "**M** = annual medication resource units."
    )

    st.subheader("Average Annual Cost by Projected State")

    cost_summary = (
        patient_data.groupby(
            "Projected_State_12m",
            dropna=False,
        )
        .agg(
            Patients=("Patient_ID", "nunique"),
            Average_Annual_Cost=(
                "Illustrative_Annual_Cost", "mean"
            ),
            Total_Annual_Cost=(
                "Illustrative_Annual_Cost", "sum"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "Projected_State_12m": "Projected State",
                "Average_Annual_Cost": (
                    "Average Annual Cost (Cost Units)"
                ),
                "Total_Annual_Cost": (
                    "Total Annual Cost (Cost Units)"
                ),
            }
        )
    )

    st.dataframe(
        cost_summary.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Patient-Level Cost Inputs")

    selected_cost_id = st.selectbox(
        "Select Patient ID",
        PATIENT_IDS,
        key="cost_patient",
    )

    row = patient_data[
        patient_data["Patient_ID"].astype(str) == selected_cost_id
    ].iloc[0]

    c1, c2 = st.columns(2)

    c1.metric(
        "Projected State",
        str(row["Projected_State_12m"]),
    )

    c2.metric(
        "Workbook Annual Cost",
        f"{row['Illustrative_Annual_Cost']:,.0f} Cost Units",
    )

    inputs = pd.DataFrame([
        {
            "Cost Input": "Annual laboratory visits (L)",
            "Value": row["Annual_Lab_Visits"],
        },
        {
            "Cost Input": "Annual outpatient visits (O)",
            "Value": row["Annual_Outpatient_Visits"],
        },
        {
            "Cost Input": "Annual hospitalizations (H)",
            "Value": row["Annual_Hospitalizations"],
        },
        {
            "Cost Input": "Medication resource units (M)",
            "Value": row["Medication_Resource_Units"],
        },
    ])

    st.dataframe(
        inputs,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "All costs are illustrative Cost Units from the synthetic dataset, "
        "not rupees, verified market prices, or official government "
        "reimbursement rates. The app displays the workbook's recorded costs."
    )


st.divider()
st.caption("Kidney Digital Twin | HEOR Module")
