import streamlit as st
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title(" Customer Conversion, Revenue Prediction & Segmentation")

@st.cache_resource
def train_models():
    X_cls, y_cls = make_classification(
        n_samples=500, n_features=3, n_informative=2, n_redundant=0,
        random_state=42
    )
    clf = LogisticRegression()
    clf.fit(X_cls, y_cls)

    X_reg, y_reg = make_regression(n_samples=500, n_features=3, noise=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)

    return clf, reg

clf_model, reg_model = train_models()
st.success(" Dummy models trained successfully!")

option = st.radio("Choose input method:", ["Upload CSV file", "Manual input"])

df = None

if option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

elif option == "Manual input":
    name = st.text_input("Name")
    age = st.number_input("Age", 0, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    annual_income = st.number_input("Annual Income", 0)

    if st.button("Submit Manual Input"):
        input_data = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Annual Income": annual_income
        }
        df = pd.DataFrame([input_data])
        st.write("### Your input:")
        st.dataframe(df)

def preprocess(df):
    df = df.copy()
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    df['Gender'] = df['Gender'].map(gender_map).fillna(-1)

    df['Age'] = df['Age'].fillna(df['Age'].median() if not df['Age'].isnull().all() else 25)
    df['Annual Income'] = df['Annual Income'].fillna(df['Annual Income'].median() if not df['Annual Income'].isnull().all() else 0)

    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    return df[['Age', 'Gender', 'Annual Income']]

# Clustering function
@st.cache_resource
def cluster_customers(df_processed, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_processed)
    return clusters, kmeans

if st.button("Run Predictions"):
    if df is None:
        st.error("Please upload a CSV file or enter manual input first!")
    else:
        try:
            df_processed = preprocess(df)

            conversion_pred = clf_model.predict(df_processed)[0]
            conversion_prob = clf_model.predict_proba(df_processed)[0][1]
            revenue_pred = reg_model.predict(df_processed)[0]

            st.subheader(" Conversion Prediction")
            st.write(f"Prediction: {'Will Convert ' if conversion_pred == 1 else 'No Convert '}")
            st.write(f"Conversion Probability: {conversion_prob:.2%}")

            st.subheader("Revenue Estimation")
            st.write(f"Estimated Revenue: ${revenue_pred:,.2f}")

            # Customer segmentation clustering
            n_clusters = st.slider("Select number of customer segments (clusters)", 2, 5, 3)
            clusters, kmeans_model = cluster_customers(df_processed, n_clusters)
            df_processed['Cluster'] = clusters

            st.subheader("📊 Customer Segments (Clusters)")
            st.write(df_processed)

            # Plot clustering scatter (Age vs Annual Income)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_processed, x='Age', y='Annual Income', hue='Cluster', palette='Set2', ax=ax)
            ax.set_title("Customer Segments by Age and Annual Income")
            st.pyplot(fig)

            # Additional visualizations
            st.subheader("📈 Visualizations")

            # Bar chart: Count by Cluster
            cluster_counts = df_processed['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)

            # Pie chart: Gender distribution in uploaded data
            if 'Gender' in df_processed.columns:
                gender_counts = df_processed['Gender'].map({0: 'Male', 1: 'Female', 2: 'Other'}).value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
                ax2.set_title("Gender Distribution")
                st.pyplot(fig2)

            # Histogram: Age distribution
            fig3, ax3 = plt.subplots()
            sns.histplot(df_processed['Age'], bins=15, kde=True, ax=ax3)
            ax3.set_title("Age Distribution")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"Prediction error: {e}")