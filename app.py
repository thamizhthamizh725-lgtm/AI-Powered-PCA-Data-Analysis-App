import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI PCA App", layout="centered")

st.title("🔍 AI-Powered PCA Data Analysis App")
st.write("Upload a CSV file to apply Principal Component Analysis (PCA)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Original Dataset")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.shape[1] < 2:
        st.warning("Dataset must have at least 2 numeric columns for PCA")
    else:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            pca_result,
            columns=['Principal Component 1', 'Principal Component 2']
        )

        st.subheader("📉 PCA Result (2D)")
        st.dataframe(pca_df.head())

        st.subheader("📊 Explained Variance Ratio")
        st.write(pca.explained_variance_ratio_)

        fig, ax = plt.subplots()
        ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Scatter Plot")

        st.pyplot(fig)

        csv = pca_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download PCA Result",
            data=csv,
            file_name="pca_result.csv",
            mime="text/csv"
        )
