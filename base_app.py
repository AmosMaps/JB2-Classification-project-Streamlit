import streamlit as st
import joblib, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the vectorizer from the pkl file
with open("vectorizer.pkl", "rb") as file:
    test_cv = pickle.load(file)

# Function to display information page
def show_information():
    st.info("In today's digital world, people read a lot more news than ever before, creating a huge amount of information for readers to sort through. To help readers find the news they care about, we aim to build a model that organizes news articles into predefined categories based on their subject matter. This will help in systematically managing and retrieving content, making it more accessible and relevant to users.")
    st.markdown("Using new technologies in machine learning and natural language processing (NLP), this project will create robust models that can automatically sort news articles into the right categories. This will lead to improved content categorization, operational efficiency, and enhanced user experience, which are crucial for a news outlet. These models will later be turned into an app for news outlets and readers to use to their advantage.")
    
# Function to display prediction page
def show_prediction():
    st.info("Prediction with ML Models")
    news_text = st.text_area("Enter Text", "Type Here")
    
    model_name = ["Logistic Regression Classifier", "Support Vector Classifier", "Random Forest Classifier"]
    model_choice = st.selectbox("Select a Classifier Model", model_name)

    if st.button("Classify"):
        vect_text = test_cv.transform([news_text])
        
        if model_choice == "Logistic Regression Classifier":
            lg = joblib.load(open(os.path.join("lg_model.pkl"), "rb"))
            prediction = lg.predict(vect_text)
        elif model_choice == "Support Vector Classifier":
            svc_classifier = joblib.load(open(os.path.join("svc_model.pkl"), "rb"))
            prediction = svc_classifier.predict(vect_text)
        elif model_choice == "Random Forest Classifier":
            rf = joblib.load(open(os.path.join("rf_model.pkl"), "rb"))
            prediction = rf.predict(vect_text)
        # Map numeric prediction to class label
        class_labels = ["Business", "Education", "Entertainment", "Sports", "Technology"]
        predicted_class = class_labels[prediction[0]]

        st.success("Text Categorized as: {}".format(predicted_class))

# Function to display "About Us" page
def show_about_us():
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Meet our team")
        st.write("##")
        st.write(
            """
            Amos Maponya - Software Engineer

            Siphosethu Rululu - Data Scientist

            Lebogang Malata - Project Manager

            Judith Kabongo - GitHub Manager

            Josephine Ndukwani - Data Scientist

            Tselane Moeti - Data Scientist
            """)

    with right_column:
        st.header("Problem statement")
        st.write("##")
        st.write(
            """The primary aim of this project is to develop and deploy an automated news article classification system using machine learning techniques. This system will be designed to accurately classify articles into predefined categories, thereby improving content organization, operational efficiency, and reader satisfaction for the news outlet.
            """)


# Main function where we will build the actual app
def main():
    """News Classifier App with Streamlit"""
    st.title("News Classifier")
    st.subheader("Analysing news articles")

    pages = [
        {"title": "Prediction", "function": show_prediction},
        {"title": "Information", "function": show_information},
        {"title": "About us", "function": show_about_us}
    ]

    page_titles = [page["title"] for page in pages]
    selection = st.sidebar.radio("Choose Option", page_titles)

    for page in pages:
        if page["title"] == selection:
            page["function"]()

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
