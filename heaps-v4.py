import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import streamlit as st





# Load trained model
MODEL_PATH = "trained-model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Define categorical mapping
category_mappings = {
    "Age": lambda x: int(0 if x <= 20 else 1 if x <= 23 else 2),
    "Sex": {"Male": 0, "Female": 1},
    "Scholarship": {"No": 0, "Yes": 1},
    "Type_of_High_School": {"Public": 0, "Private": 1},
    "SHS_GWA": lambda x: int(0 if x >= 90 else 1),
    "Entrance_Exam_Result": {"1st qualifier": 0, "2nd qualifier": 1},
    "Study_Hours": {"Less than 1 hour": 0, "2-3 hours": 1, "4-5 hours": 2, "More than 5 hours": 3},
    "Submission_Activities": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3},
    "Consultation": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Always": 3},
    "Attendance": {"Below 30 days (<50%)": 0, "44-30 days (50-69%)": 1, "59-45 days (70-89%)": 2, "70-60 days (90-100%)": 3},
    "Part_Time_Job": {"No": 0, "Yes": 1},
    "Devices": {"No": 0, "Yes": 1},
    "Internet_Access": {"No": 0, "Yes": 1},
    "Daily_Allowance": {"Below Php 50": 0, "Php 50-100": 1, "Php 101-200": 2, "More than Php 200": 3},
}

def preprocess_data(df):
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

st.set_page_config(page_title="HEAPS v2.8", layout="wide")

# Custom styling with header and footer
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #007BFF; color: white; border-radius: 5px; padding: 10px 20px;}
    .stDownloadButton>button {background-color: #28A745; color: white; border-radius: 5px; padding: 10px 20px;}
    .sidebar .sidebar-content {background-color: #2c3e50; color: white;}
    .sidebar .sidebar-content a {color: white;}
    .header {background-color: #007BFF; padding: 15px; color: white; text-align: center; font-size: 20px; font-weight: bold;}
    .footer {background-color: #2c3e50; padding: 10px; color: white; text-align: center; font-size: 14px; margin-top: 20px;}
    </style>
    <div class='header'>Higher Education Student Academic Prediction System</div>
    """, unsafe_allow_html=True)

# Store multiple datasets
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = {}

st.title("📊 HEAPS")



# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Preprocess Data", "Predictions", "Visualizations", "Generate Report"])

if page == "Home":
    st.write(
        "This web application serves as a tool which helps universities and colleges predict student academic risk using machine learning models. Upload student data, preprocess it, generate predictions, and visualize insights.")
    st.image("assets/heaps.jpg", caption="Empowering Education with Data-Driven Insights", use_container_width=True)

    st.markdown("### Key Features:")
    st.markdown("✔ Upload and preprocess student data")
    st.markdown("✔ Predict student academic risk")
    st.markdown("✔ Visualize insights with charts and graphs")
    st.markdown("✔ Generate and export reports")

elif page == "Upload & Preprocess Data":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.dataframe(df.head())
        st.write(f"Total Rows in Uploaded File: {df.shape[0]}")

        course = st.selectbox("Select Course", ["BSAB", "BSCrim", "BSIT", "BEED", "BSED", "BSHM", "Other"])
        if course == "Other":
            course = st.text_input("Enter Course Name")
        year = st.selectbox("Select Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
        section = st.text_input("Enter Section")

        if st.button("Preprocess Data"):
            df = preprocess_data(df)
            st.session_state["processed_data"] = df
            st.session_state["course"] = course
            st.session_state["year"] = year
            st.session_state["section"] = section
            st.success("Data Preprocessed Successfully!")
            st.dataframe(df.head())

elif page == "Predictions":
    st.write("### Predict Student Risk")
    st.write("Upload and preprocess the dataset first. Then click the 'Predict' button to generate results.")
    if "processed_data" not in st.session_state:
        st.error("Please upload and preprocess a dataset first before making predictions.")
    else:
        if st.button("Predict"):
            st.write("Processing prediction... Please wait.")
            df = st.session_state["processed_data"].copy()
            if model:
                predictions = model.predict(df)
                df["Risk_Status"] = ["At Risk" if p == 1 else "Low Risk" for p in predictions]
                st.session_state["prediction_results"] = df
                st.success("Prediction completed!")
                st.dataframe(df.style.set_properties(**{'background-color': '#f0f0f0', 'color': 'black'}))
elif page == "Visualizations":
    st.write("### Visualization of Predictions")
    if "prediction_results" not in st.session_state:
        st.error("Please complete predictions first.")
    else:
        df = st.session_state["prediction_results"]
        viz_option = st.selectbox("Select Visualization Type:",
                                  ["Pie Chart", "Bar Chart", "Feature Importance", "Histogram"])
        if viz_option == "Pie Chart":
            fig = px.pie(df, names="Risk_Status", title="Risk Status Distribution")
        elif viz_option == "Bar Chart":
            fig = px.bar(df, x="Risk_Status", title="Count of Students by Risk Category", color="Risk_Status")
        elif viz_option == "Feature Importance":
            if model:
                importances = model.feature_importances_
                features = df.columns[:-1]
                fig = px.bar(x=features, y=importances, title="Feature Importance",
                             labels={'x': 'Features', 'y': 'Importance'})
        elif viz_option == "Histogram":
            feature = st.selectbox("Select a feature:", df.columns[:-1])
            fig = px.histogram(df, x=feature, title=f"Histogram of {feature}")
        st.plotly_chart(fig)


elif page == "Generate Report":
    if "prediction_results" not in st.session_state:
        st.error("Please complete predictions first.")
    else:
        st.write("### Report Generation")
        header = st.text_input("Edit Report Header", "Student Risk Report")
        footer = st.text_input("Edit Report Footer", "End of Report")
        if st.button("Generate Report"):
            st.success(f"Report generated for {st.session_state['course']} - {st.session_state['year']} - Section {st.session_state['section']}")
# Footer
st.markdown("""
    <div class='footer'>
    Developed for Central Philippines State University. All Rights Reserved.
    </div>
    """, unsafe_allow_html=True)