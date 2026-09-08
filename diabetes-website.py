import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_risk.csv")
    return df

data = load_data()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🩺 Diabetes Risk Analysis Dashboard")
st.markdown("### Exploratory Data Analysis using Streamlit")

st.divider()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("🔎 Filters")

# Gender
gender_options = data["gender"].dropna().unique().tolist()

selected_gender = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

# City
city_options = data["city"].dropna().unique().tolist()

selected_city = st.sidebar.multiselect(
    "City",
    city_options,
    default=city_options
)

# Diabetes Risk
risk_options = data["diabetes_risk"].dropna().unique().tolist()

selected_risk = st.sidebar.multiselect(
    "Diabetes Risk",
    risk_options,
    default=risk_options
)

# Physical Activity
activity_options = data["physical_activity_level"].dropna().unique().tolist()

selected_activity = st.sidebar.multiselect(
    "Physical Activity",
    activity_options,
    default=activity_options
)

# ---------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------

filtered_data = data[
    (data["gender"].isin(selected_gender)) &
    (data["city"].isin(selected_city)) &
    (data["diabetes_risk"].isin(selected_risk)) &
    (data["physical_activity_level"].isin(selected_activity))
]

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Patients",
        len(filtered_data)
    )

with col2:
    st.metric(
        "Average Age",
        round(filtered_data["age"].mean(), 1)
    )

with col3:
    st.metric(
        "Average BMI",
        round(filtered_data["bmi"].mean(), 1)
    )

with col4:
    st.metric(
        "High Risk Patients",
        len(
            filtered_data[
                filtered_data["diabetes_risk"] == "High"
            ]
        )
    )

st.divider()

# ---------------------------------------------------
# DIABETES RISK DISTRIBUTION
# ---------------------------------------------------

st.subheader("🎯 Diabetes Risk Distribution")

col1, col2 = st.columns(2)

with col1:

    risk_counts = filtered_data["diabetes_risk"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=filtered_data,
        x="diabetes_risk",
        ax=ax
    )

    ax.set_title("Diabetes Risk Distribution")
    ax.set_xlabel("Diabetes Risk")
    ax.set_ylabel("Number of Patients")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Diabetes Risk Percentage")

    st.pyplot(fig)

# ---------------------------------------------------
# CATEGORICAL ANALYSIS
# ---------------------------------------------------

st.divider()

st.subheader("📊 Categorical Analysis")

categorical_columns = [
    "gender",
    "city",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "income_bracket"
]

selected_category = st.selectbox(
    "Select categorical variable",
    categorical_columns
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(
    data=filtered_data,
    x=selected_category,
    hue="diabetes_risk",
    ax=ax
)

ax.set_title(
    f"{selected_category} vs Diabetes Risk"
)

ax.set_xlabel(selected_category)
ax.set_ylabel("Number of Patients")

plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------------------------------------------
# NUMERICAL ANALYSIS
# ---------------------------------------------------

st.divider()

st.subheader("📈 Numerical Analysis")

numerical_columns = [
    "age",
    "bmi",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm"
]

selected_numeric = st.selectbox(
    "Select numerical variable",
    numerical_columns
)

chart_type = st.radio(
    "Select chart type",
    ["Histogram", "Boxplot", "Scatter Plot"],
    horizontal=True
)

if chart_type == "Histogram":

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=filtered_data,
        x=selected_numeric,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {selected_numeric}"
    )

    st.pyplot(fig)


elif chart_type == "Boxplot":

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=filtered_data,
        x="diabetes_risk",
        y=selected_numeric,
        ax=ax
    )

    ax.set_title(
        f"{selected_numeric} by Diabetes Risk"
    )

    st.pyplot(fig)


elif chart_type == "Scatter Plot":

    second_numeric = st.selectbox(
        "Select second numerical variable",
        [
            col for col in numerical_columns
            if col != selected_numeric
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=filtered_data,
        x=selected_numeric,
        y=second_numeric,
        hue="diabetes_risk",
        ax=ax
    )

    ax.set_title(
        f"{selected_numeric} vs {second_numeric}"
    )

    st.pyplot(fig)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

st.divider()

st.subheader("🔥 Correlation Heatmap")

correlation = filtered_data[numerical_columns].corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

ax.set_title("Correlation Between Numerical Variables")

st.pyplot(fig)

# ---------------------------------------------------
# FAMILY HISTORY ANALYSIS
# ---------------------------------------------------

st.divider()

st.subheader("🧬 Family History vs Diabetes Risk")

fig, ax = plt.subplots(figsize=(9, 5))

sns.countplot(
    data=filtered_data,
    x="family_history_diabetes",
    hue="diabetes_risk",
    ax=ax
)

ax.set_title(
    "Family History of Diabetes vs Diabetes Risk"
)

ax.set_xlabel("Family History of Diabetes")
ax.set_ylabel("Number of Patients")

st.pyplot(fig)

# ---------------------------------------------------
# PHYSICAL ACTIVITY ANALYSIS
# ---------------------------------------------------

st.subheader("🏃 Physical Activity vs Diabetes Risk")

fig, ax = plt.subplots(figsize=(9, 5))

sns.countplot(
    data=filtered_data,
    x="physical_activity_level",
    hue="diabetes_risk",
    ax=ax
)

ax.set_title(
    "Physical Activity Level vs Diabetes Risk"
)

ax.set_xlabel("Physical Activity Level")
ax.set_ylabel("Number of Patients")

st.pyplot(fig)

# ---------------------------------------------------
# SMOKING ANALYSIS
# ---------------------------------------------------

st.subheader("🚬 Smoking Status vs Diabetes Risk")

fig, ax = plt.subplots(figsize=(9, 5))

sns.countplot(
    data=filtered_data,
    x="smoking_status",
    hue="diabetes_risk",
    ax=ax
)

ax.set_title(
    "Smoking Status vs Diabetes Risk"
)

ax.set_xlabel("Smoking Status")
ax.set_ylabel("Number of Patients")

plt.xticks(rotation=30)

st.pyplot(fig)

# ---------------------------------------------------
# DATA TABLE
# ---------------------------------------------------

st.divider()

st.subheader("📋 Patient Data")

st.dataframe(
    filtered_data,
    use_container_width=True
)

# ---------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------

csv = filtered_data.to_csv(index=False)

st.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name="filtered_diabetes_risk.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Diabetes Risk Analysis Dashboard | Built with Python, Pandas, Seaborn and Streamlit"
)