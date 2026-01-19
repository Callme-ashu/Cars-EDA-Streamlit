import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cars EDA Project 🚗", layout="wide")


@st.cache_data
def load_raw():
    return pd.read_csv("Cars.csv")

@st.cache_data
def load_cleaned():
    return pd.read_csv("Cars_cleaned.csv")

raw = load_raw()
clean = load_cleaned()

page = st.sidebar.radio("🧭 Navigation",
["🚘 Introduction","📊 Analysis","📌 Conclusions"])


# ===================== INTRODUCTION PAGE =====================
if page == "🚘 Introduction":

    st.title("🚗 Cars Analytics Dashboard")

    st.markdown("""
    ## 🔍 Introduction

    Exploratory Data Analysis (EDA) is a crucial step in any data science or data analytics project, 
    as it helps in understanding the structure, patterns, and hidden insights present within the dataset 
    before applying any advanced modeling techniques.

    In this project, we perform an in-depth **Exploratory Data Analysis on a Cars dataset**, 
    which contains detailed information about used cars available in the market.

    The primary objective of this project is to **analyze various factors that influence car prices and 
    consumer preferences**, such as brand, manufacturing year, fuel type, mileage, engine capacity, 
    transmission type, and ownership history.

    The dataset initially contains raw and unprocessed information, including missing values and 
    inconsistent formats. Therefore, the project starts with **data cleaning and preprocessing**.

    After cleaning, **univariate, bivariate, and multivariate analysis** is performed using 
    meaningful visualizations to uncover insights and market trends.

    This EDA creates a strong analytical foundation for future tasks such as 
    **car price prediction and recommendation systems 🚀**.
    """)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("🚘 Total Cars", len(clean))
    c2.metric("💰 Average Price", round(clean["Price"].mean(),2))
    c3.metric("📏 Average KM", int(clean["Kilometers_Driven"].mean()))
    c4.metric("🏭 Total Companies", clean["Company_Name"].nunique())


    st.subheader("📄 Raw Dataset")
    st.dataframe(raw, use_container_width=True)

    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(clean, use_container_width=True)


    st.subheader("🗺️ Location Map")

    if "Latitude" in clean.columns and "Longitude" in clean.columns:
        st.map(clean[["Latitude","Longitude"]])
    else:
        st.info("ℹ️ Latitude and Longitude not available")


# ===================== ANALYSIS PAGE =====================
elif page == "📊 Analysis":

    st.title("📊 Exploratory Analysis Studio")

    company = st.sidebar.multiselect(
        "🏢 Select Company",
        options=clean["Company_Name"].unique(),
        default=clean["Company_Name"].unique()
    )

    year = st.sidebar.slider(
        "📅 Select Year Range",
        int(clean["Year"].min()),
        int(clean["Year"].max()),
        (int(clean["Year"].min()), int(clean["Year"].max()))
    )

    df = clean[(clean["Company_Name"].isin(company)) &
               (clean["Year"].between(year[0],year[1]))]


    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()


    k1,k2,k3 = st.columns(3)

    k1.metric("🚗 Selected Cars", len(df))
    k2.metric("💸 Average Price", round(df["Price"].mean(),2))
    k3.metric("⚡ Average Power", round(df["Power_value"].mean(),2))


    st.header("📈 Univariate Analysis")

    col = st.selectbox("🔎 Choose Column", df.columns)

    fig, ax = plt.subplots(figsize=(7,4))

    if col in cat_cols:
        sns.countplot(y=df[col], ax=ax)

    else:
        dist = st.radio("📊 View Type",["Histogram","KDE","Boxplot"])

        if dist=="Histogram":
            sns.histplot(df[col], kde=True, ax=ax)

        elif dist=="KDE":
            sns.kdeplot(df[col], fill=True, ax=ax)

        else:
            sns.boxplot(x=df[col], ax=ax)

    st.pyplot(fig)


    st.header("🔁 Bivariate Analysis")

    c1,c2 = st.columns(2)

    x = c1.selectbox("📐 X Axis", df.columns)
    y = c2.selectbox("📏 Y Axis", df.columns)

    fig2,ax2 = plt.subplots(figsize=(7,4))

    if x in num_cols and y in num_cols:
        sns.scatterplot(data=df,x=x,y=y,ax=ax2)
        st.write("📉 Correlation:", round(df[x].corr(df[y]),3))

    elif x in num_cols and y in cat_cols:
        sns.boxplot(data=df,x=y,y=x,ax=ax2)

    elif x in cat_cols and y in num_cols:
        sns.boxplot(data=df,x=x,y=y,ax=ax2)

    else:
        sns.countplot(data=df,x=x,hue=y,ax=ax2)

    st.pyplot(fig2)


    st.header("🧠 Multivariate Analysis")

    option = st.selectbox("🛠️ Method",
    ["Heatmap","Pairplot","Grouped Bar"])


    if option=="Heatmap":
        fig3,ax3 = plt.subplots(figsize=(9,5))
        sns.heatmap(df[num_cols].corr(),annot=True,cmap="coolwarm",ax=ax3)
        st.pyplot(fig3)

    elif option=="Pairplot":
        pair = sns.pairplot(df[num_cols])
        st.pyplot(pair)

    else:
        if "Fuel_Type" in df.columns and "Price" in df.columns:

            fig4,ax4 = plt.subplots(figsize=(8,4))

            sns.barplot(
                data=df,
                x="Fuel_Type",
                y="Price",
                hue="Transmission" if "Transmission" in df.columns else None,
                ax=ax4
            )

            st.pyplot(fig4)

        else:
            st.warning("⚠️ Required columns missing")


# ===================== CONCLUSION PAGE =====================
else:

    st.title("📌 Automated Insights")

    st.markdown("""
    ## 📊 Conclusion

    In this Exploratory Data Analysis project, we successfully analyzed the Cars dataset to 
    extract meaningful insights about car pricing and market behavior.

    After cleaning and preprocessing the data, we explored the distribution and relationships 
    between key features such as price, year, fuel type, brand, and transmission.

    The analysis revealed that:
    - 🚗 Newer cars generally have higher prices  
    - 🏷️ Brand reputation significantly impacts car value  
    - ⚙️ Transmission type and fuel type also influence pricing  

    Visualizations helped simplify complex data patterns and provided a clear understanding 
    of trends and relationships.

    This project highlights the importance of EDA in transforming raw data into actionable insights. 
    The cleaned dataset and analysis can further be used for 
    **machine learning models such as car price prediction 🤖**.
    """)

    st.write("📦 Total Records:", len(clean))

    st.write("🏆 Highest Price Car:",
             clean.loc[clean["Price"].idxmax(),"Company_Name"])

    st.write("⛽ Most Common Fuel:",
             clean["Fuel_Type"].mode()[0])

    st.write("📈 Strongest Correlation with Price:",
             clean.select_dtypes(include=np.number)
             .corr()["Price"].sort_values(ascending=False).index[1])

    st.success("🎉 Thanks for visiting this Cars EDA Project 🚗")

