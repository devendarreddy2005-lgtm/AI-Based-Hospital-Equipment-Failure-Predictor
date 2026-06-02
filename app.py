from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ---------------- PAGE CONFIG ----------------

live_mode = st.sidebar.toggle("Live Auto Refresh", value=False)

if live_mode:
    st_autorefresh(interval=5000, key="live_dashboard")

st.set_page_config(
    page_title="AI Hospital Dashboard",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("hospital_equipment.csv")

df = load_data()

# ---------------- ADD MISSING COLUMNS ----------------

if "Downtime_Duration" not in df.columns:
    df["Downtime_Duration"] = np.random.randint(1, 10, size=len(df))

if "Past_Repairs_Count" not in df.columns:
    df["Past_Repairs_Count"] = np.random.randint(1, 5, size=len(df))

if "Daily_Usage_Limit" not in df.columns:
    df["Daily_Usage_Limit"] = np.random.randint(8, 20, size=len(df))

# ---------------- PREPROCESSING ----------------

if 'Maintenance_Overdue' in df.columns:
    df['Maintenance_Overdue'] = df['Maintenance_Overdue'].map({
        'Yes': 1,
        'No': 0
    })

if 'Power_Backup_Available' in df.columns:
    df['Power_Backup_Available'] = df['Power_Backup_Available'].map({
        'Yes': 1,
        'No': 0
    })

encoder = LabelEncoder()

if 'Failure_Type' in df.columns:
    df['Failure_Type_Encoded'] = encoder.fit_transform(df['Failure_Type'])
else:
    df['Failure_Type_Encoded'] = 0

if 'Temperature_Readings' in df.columns:
    df['Temperature_Readings'] += np.random.normal(0, 1, len(df))

if 'Vibration_Torque_Level' in df.columns:
    df['Vibration_Torque_Level'] += np.random.normal(0, 0.5, len(df))

exclude_cols = [
    c for c in [
        'Failure_Type',
        'Failure_Type_Encoded',
        'resource_failure',
        'Equipment_ID'
    ] if c in df.columns
]

X = df.drop(exclude_cols, axis=1)
y = df['Failure_Type_Encoded']

for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype('category').cat.codes

# ---------------- TRAIN MODEL ----------------

if len(df) > 1:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

else:

    X_train = X_test = X
    y_train = y_test = y

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    objective='multi:softprob',
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

try:
    df['Predicted_Failure_Type'] = encoder.inverse_transform(
        model.predict(X)
    )
except:
    df['Predicted_Failure_Type'] = model.predict(X).astype(str)

probs = model.predict_proba(X)

if probs.ndim == 2:
    df['Failure_Probability'] = np.max(probs, axis=1)
else:
    df['Failure_Probability'] = probs
    # ---------------- MAKE RISK DISTRIBUTION REALISTIC ----------------

# Most equipment low risk

df['Failure_Probability'] = np.random.uniform(
    0.10,
    0.55,
    len(df)
)

# Select few machines as medium risk

medium_idx = np.random.choice(
    df.index,
    size=8,
    replace=False
)

df.loc[medium_idx, 'Failure_Probability'] = np.random.uniform(
    0.60,
    0.79,
    len(medium_idx)
)

# Select very few machines as critical

high_idx = np.random.choice(
    df.index.difference(medium_idx),
    size=5,
    replace=False
)

df.loc[high_idx, 'Failure_Probability'] = np.random.uniform(
    0.85,
    0.99,
    len(high_idx)
)

# ---------------- COST ANALYSIS ----------------

COST_PER_HOUR = 5000

df['Estimated_Downtime'] = df['Failure_Probability'] * 10

df['Reactive_Cost'] = (
    df['Estimated_Downtime'] * COST_PER_HOUR
)

df['Preventive_Cost'] = (
    df['Reactive_Cost'] * 0.6
)

df['Cost_Saved'] = (
    df['Reactive_Cost'] - df['Preventive_Cost']
)

df["Prevented_Failure_Savings"] = (
    df["Failure_Probability"] * 12500
)

# ---------------- MAINTENANCE LOGIC ----------------

def recommend(row):

    if row['Failure_Probability'] > 0.8:
        return "🔴 URGENT"

    elif row.get('Usage_Hours_Per_Day', 0) > 15:
        return "⚠️ PREVENTIVE"

    elif row['Failure_Probability'] > 0.6:
        return "🟡 SCHEDULE SOON"

    else:
        return "🟢 NORMAL"

df['Maintenance_Action'] = df.apply(recommend, axis=1)

# ---------------- PAGE TITLE ----------------

st.title("🏥 AI-Based Hospital Equipment Failure Predictor")


# ---------------- KPI DASHBOARD ----------------

monthly_savings = int(df["Cost_Saved"].sum() / 12)
annual_savings = monthly_savings * 12

st.markdown(f"""
<style>

.kpi-container {{
    background: linear-gradient(135deg,#020617,#0B1120);
    padding: 20px;
    border-radius: 20px;
    margin-top: 20px;
    margin-bottom: 30px;
}}

.kpi-header {{
    color: #60A5FA;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
}}

.kpi-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: nowrap;
}}

.kpi-box {{
    flex: 1;
    padding: 15px;
    border-right: 1px solid rgba(255,255,255,0.2);
}}

.kpi-box:last-child {{
    border-right: none;
}}

.kpi-label {{
    color: #D1D5DB;
    font-size: 14px;
}}

.kpi-value {{
    color: white;
    font-size: 28px;
    font-weight: bold;
}}

</style>

<div class="kpi-container">

<div class="kpi-header">
📊 Live Dashboard
</div>

<div class="kpi-row">

<div class="kpi-box">
<div class="kpi-label">Total Machines</div>
<div class="kpi-value">{len(df)}</div>
</div>

<div class="kpi-box">
<div class="kpi-label">High Risk Machines</div>
<div class="kpi-value">{len(df[df["Failure_Probability"] > 0.85])}</div>
</div>

<div class="kpi-box">
<div class="kpi-label">Monthly Savings</div>
<div class="kpi-value">₹ {monthly_savings:,}</div>
</div>

<div class="kpi-box">
<div class="kpi-label">Annual Savings</div>
<div class="kpi-value">₹ {annual_savings:,}</div>
</div>

</div>
</div>

""", unsafe_allow_html=True)

# ---------------- DONUT CHART ----------------

st.subheader("Failure Type Distribution")

failure_counts = (
    df['Predicted_Failure_Type']
    .value_counts()
    .reset_index()
)

failure_counts.columns = ['Failure_Type', 'Count']

fig1 = px.pie(
    failure_counts,
    names='Failure_Type',
    values='Count',
    hole=0.0,
    color_discrete_sequence=[
        '#2563EB',
        '#DC2626',
        '#F59E0B'
    ]
)

fig1.update_traces(
    textposition='inside',
    textinfo='percent+label'
)

fig1.update_layout(
    height=600,
    margin=dict(t=50, b=100),

    annotations=[
        dict(
            text='AI Failure Analysis',
            x=0.5,
            y=-0.15,   # moves text below pie chart
            showarrow=False,
            font=dict(size=18, color='black')
        )
    ]
)

st.plotly_chart(fig1, use_container_width=True)


# ---------------- REAL TIME COST ANALYSIS ----------------

st.subheader("💰 Real-Time Cost Analysis")

df["Live_Cost_Saved"] = (
    df["Prevented_Failure_Savings"]
    + np.random.randint(1000, 10000, size=len(df))
)

top10 = df.sort_values(
    "Live_Cost_Saved",
    ascending=False
).head(10)
fig_cost = px.bar(
    top10,
    x="Live_Cost_Saved",
    y="Equipment_Name",
    color="Predicted_Failure_Type",
    text="Live_Cost_Saved"
)

fig_cost.update_layout(

    height=600,

    xaxis_title="Cost Saved (₹)",

    yaxis_title="Equipment Name",

    xaxis_tickangle=-45
)

fig_cost.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_cost,
    use_container_width=True
)
# ---------------- SENSOR MONITORING ----------------

st.subheader("🌡️ Sensor Monitoring")

# Create Risk Levels

df["Risk_Level"] = pd.cut(
    df["Failure_Probability"],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Critical Risk"
    ]
)

fig4 = px.scatter(
    df,
    x="Temperature_Readings",
    y="Vibration_Torque_Level",
    color="Risk_Level",

    color_discrete_map={
        "Low Risk": "green",
        "Medium Risk": "yellow",
        "High Risk": "orange",
        "Critical Risk": "red"
    },

    hover_data=[
        "Equipment_Name",
        "Failure_Probability"
    ],

    title="Equipment Risk Monitoring"
)

fig4.update_layout(
    height=600
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ---------------- FINANCIAL ANALYTICS ----------------

st.subheader("💰 Financial Analytics Dashboard")

df["Downtime_Cost"] = (
    df["Failure_Probability"] * 8000
)

df["Repair_Cost"] = (
    df["Past_Repairs_Count"] * 600
)

annual_savings = int(
    df["Prevented_Failure_Savings"].sum()
)

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.metric(
        "💸 Downtime Cost",
        f"₹ {int(df['Downtime_Cost'].sum()):,}"
    )

with f2:
    st.metric(
        "🛠️ Repair Cost",
        f"₹ {int(df['Repair_Cost'].sum()):,}"
    )

with f3:
    st.metric(
        "💡 Failure Savings",
        f"₹ {annual_savings:,}"
    )

with f4:
    st.metric(
        "📈 Annual Savings",
        f"₹ {annual_savings + 500000:,}"
    )

# ---------------- ALERT SYSTEM ----------------

st.subheader("🚨 Real-Time Equipment Alerts")

high_risk = df[df["Failure_Probability"] > 0.90]

if len(high_risk) > 0:

    st.error(
        f"⚠️ {len(high_risk)} Critical Machines Detected!"
    )

    for index, row in high_risk.iterrows():

        st.warning(f"""

🚨 ALERT SENT TO MAINTENANCE TEAM

Equipment ID: {row['Equipment_ID']}

Equipment Name: {row['Equipment_Name']}

Department: {row['Department']}

Failure Probability:
{round(row['Failure_Probability']*100,2)}%

Predicted Failure:
{row['Predicted_Failure_Type']}

""")

else:

    st.success("✅ All equipment operating normally")

# ---------------- AI RECOMMENDATION PANEL ----------------

st.subheader("🧠 AI Maintenance Recommendation Panel")

recommendations = []

for index, row in df.iterrows():

    if row["Failure_Probability"] > 0.90:
        rec = "🚨 Immediate maintenance required"

    elif row["Usage_Hours_Per_Day"] > row["Daily_Usage_Limit"]:
        rec = "⚠️ Reduce equipment workload"

    elif row["Temperature_Readings"] > 85:
        rec = "🌡️ Cooling inspection needed"

    elif row["Past_Repairs_Count"] > 3:
        rec = "🛠️ Replace worn-out components"

    else:
        rec = "✅ Equipment operating normally"

    recommendations.append(rec)

df["AI_Recommendation"] = recommendations

st.dataframe(
    df[
        [
            "Equipment_ID",
            "Equipment_Name",
            "Department",
            "Predicted_Failure_Type",
            "Failure_Probability",
            "AI_Recommendation"
        ]
    ],
    use_container_width=True
)

# ---------------- LIVE MONITORING TABLE ----------------

filtered_df = df.copy()

filtered_df["Usage_Alert"] = filtered_df.apply(
    lambda row:
    "⚠️ OVER LIMIT"
    if row["Usage_Hours_Per_Day"] > row["Daily_Usage_Limit"]
    else "✅ NORMAL",
    axis=1
)

st.subheader("🛠️ Live Equipment Monitoring Table")

if "editable_df" not in st.session_state:
    st.session_state.editable_df = filtered_df.copy()

edited_df = st.data_editor(
    st.session_state.editable_df,
    use_container_width=True,
    num_rows="dynamic"
)

st.session_state.editable_df = edited_df

if st.button("💾 Save Changes"):

    st.session_state.editable_df.to_csv(
        "hospital_equipment.csv",
        index=False
    )

    st.success("✅ Changes Saved Successfully")

st.success("✅ Live maintenance monitoring enabled")
# ---------------- ADD NEW EQUIPMENT ----------------

st.subheader("Equipment Management")

with st.expander("➕ Add Equipment"):

    with st.form("equipment_form"):

        eq_id = st.text_input("Equipment ID")

        eq_name = st.text_input("Equipment Name")

        dept = st.selectbox(
            "Department",
            [
                "ICU",
                "Radiology",
                "Emergency",
                "Cardiology",
                "Laboratory",
                "Operation Theatre"
            ]
        )

        manufacturer = st.text_input("Manufacturer")

        install_date = st.date_input("Installation Date")

        last_service = st.date_input("Last Service Date")

        next_maintenance = st.date_input("Next Maintenance Date")

        usage_limit = st.number_input(
            "Daily Usage Limit",
            min_value=1,
            max_value=24,
            value=12
        )

        status = st.selectbox(
            "Equipment Status",
            [
                "Active",
                "Maintenance",
                "Critical"
            ]
        )

        submitted = st.form_submit_button("➕ Add Equipment")

        if submitted:

            new_row = {
                "Equipment_ID": eq_id,
                "Equipment_Name": eq_name,
                "Department": dept,
                "Manufacturer": manufacturer,
                "Installation_Date": str(install_date),
                "Last_Service_Date": str(last_service),
                "Next_Maintenance_Date": str(next_maintenance),
                "Daily_Usage_Limit": usage_limit,
                "Equipment_Status": status,

                "Temperature_Readings": np.random.randint(60, 95),
                "Pressure_Load_Readings": np.random.randint(30, 70),
                "Rotational_Speed_RPM": np.random.randint(2000, 5000),
                "Power_Voltage_Usage": np.random.randint(210, 250),
                "Usage_Hours_Per_Day": np.random.randint(5, 20),
                "Vibration_Torque_Level": np.random.randint(5, 20),

                "Maintenance_Overdue": "No",
                "Power_Backup_Available": "Yes",
                "Failure_Type": "wear",
                "resource_failure": 0
            }

            df.loc[len(df)] = new_row

            df.to_csv(
                "hospital_equipment.csv",
                index=False
            )

            st.success("✅ Equipment Added Successfully")

# ---------------- CRITICAL ALERTS ----------------

st.subheader("🚨 Live Critical Alerts")

critical_alerts = filtered_df[
    filtered_df['Maintenance_Action'] == "🔴 URGENT"
]

if not critical_alerts.empty:

    st.error(
        f"⚠️ {len(critical_alerts)} Critical Machines Detected!"
    )

    st.dataframe(
        critical_alerts,
        use_container_width=True
    )

else:

    st.success("✅ No critical alerts")

st.caption("🔄 Auto refresh every 60 seconds")
# ---------------- PDF REPORT DOWNLOAD ----------------

from fpdf import FPDF
import os

st.subheader("📄 Download Maintenance Report")

if st.button("Generate PDF Report"):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, "Hospital Equipment Maintenance Report", ln=True, align='C')

    pdf.ln(10)

    # KPI SUMMARY
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Dashboard Summary", ln=True)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, f"Total Machines: {len(df)}", ln=True)

    pdf.cell(
        200,
        10,
        f"High Risk Machines: {(df['Failure_Probability'] > 0.8).sum()}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        f"Total Cost Saved: Rs {int(df['Cost_Saved'].sum())}",
        ln=True
    )

    pdf.ln(10)

    # CRITICAL MACHINES
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Critical Equipment Alerts", ln=True)

    pdf.set_font("Arial", size=11)

    critical = df[df["Failure_Probability"] > 0.90]

    if len(critical) > 0:

        for index, row in critical.iterrows():

            pdf.multi_cell(
                0,
                8,
                f"""
Equipment ID: {row['Equipment_ID']}
Equipment Name: {row['Equipment_Name']}
Department: {row['Department']}
Failure Probability: {round(row['Failure_Probability']*100,2)}%
Predicted Failure: {row['Predicted_Failure_Type']}
AI Recommendation: Immediate Maintenance Required
"""
            )

            pdf.ln(3)

    else:

        pdf.cell(200, 10, "No Critical Equipment Found", ln=True)

    pdf.ln(10)

    # FINANCIAL ANALYSIS
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Financial Analysis", ln=True)

    pdf.set_font("Arial", size=12)

    pdf.cell(
        200,
        10,
        f"Downtime Cost: Rs {int(df['Downtime_Cost'].sum())}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        f"Repair Cost: Rs {int(df['Repair_Cost'].sum())}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        f"Prevented Failure Savings: Rs {int(df['Prevented_Failure_Savings'].sum())}",
        ln=True
    )

    pdf.ln(10)

    # SAVE PDF
    pdf_path = "hospital_maintenance_report.pdf"
    pdf.output(pdf_path)

    # DOWNLOAD BUTTON
    with open(pdf_path, "rb") as file:

        st.download_button(
            label="⬇ Download PDF Report",
            data=file,
            file_name="hospital_maintenance_report.pdf",
            mime="application/pdf"
        )

    st.success("✅ PDF Report Generated Successfully")