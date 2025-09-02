import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Insights App", layout="wide")

st.title("Customer Insights: Classification | Regression | Clustering")

#  File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df2 = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.write(df2.head())
    st.write("Columns in dataset:", df2.columns.tolist())

    st.subheader(" Classification Example")
    possible_targets = [col for col in df2.columns if df2[col].nunique() <= 10 and df2[col].dtype != float]
    
    if possible_targets:
        target_col = st.selectbox("Select target column for classification:", possible_targets)
        df_class = df2.dropna(subset=[target_col])
        X = df_class.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
        y = df_class[target_col]

        if not X.empty and len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, y_train)
            st.write("Classification Accuracy:", clf.score(X_test, y_test))
            st.write("Sample Predictions:", clf.predict(X_test)[:10])
        else:
            st.warning("Not enough numeric features or only one class present.")
    else:
        st.warning("No suitable categorical target columns found for classification.")


    # Regression
    # -----------------------
    if 'price' in df2.columns:
        st.subheader(" Regression Example")
        df_reg = df2.dropna(subset=['price'])
        X = df_reg.select_dtypes(include=np.number).drop(columns=['price'], errors='ignore')
        y = df_reg['price']

    if not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        st.write("Regression R² Score:", reg.score(X_test, y_test))
        st.write("Sample Predictions:", reg.predict(X_test)[:10])
    else:
        st.warning("Not enough numeric features for regression.")

    # -----------------------
    #  Clustering
    # -----------------------
    st.subheader("📌 Clustering Example")
    num_df = df2.select_dtypes(include=np.number)
    if not num_df.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(num_df.fillna(0))
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df2['Cluster'] = clusters
        st.write("Cluster Counts:", df2['Cluster'].value_counts())

        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(num_df.iloc[:, 0], num_df.iloc[:, 1], c=clusters, cmap='viridis')
            ax.set_xlabel(num_df.columns[0])
            ax.set_ylabel(num_df.columns[1])
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)
    else:
        st.warning("No numeric data for clustering.")

    # -----------------------
    #  Visualizations
    # -----------------------
    st.subheader("📊 Data Visualizations")

    col1, col2, col3 = st.columns(3)

    # Bar Chart
    with col1:
        if 'country' in df2.columns and 'price 2' in df2.columns:
            fig, ax = plt.subplots()
            df2.groupby('country')['price 2'].mean().plot(kind='bar', ax=ax)
            ax.set_title("Average price by Country")
            ax.set_ylabel("Average price")
            ax.set_xlabel("Country")
            plt.xticks(rotation=45)
            st.pyplot(fig)


    # Histogram -
    with col2:
        if 'price' in df2.columns:
            fig, ax = plt.subplots()
            df2['price'].hist(ax=ax, bins=20, edgecolor='black')
            ax.set_title("Price Distribution")
            ax.set_xlabel("Price")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Pair Plot 
    with col3:
        num_cols = df2.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            st.write("Pair Plot (first 4 numeric columns)")
            fig = sns.pairplot(df2[num_cols[:4]])
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start.")