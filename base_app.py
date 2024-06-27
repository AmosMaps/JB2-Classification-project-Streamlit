"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load your vectorizer from the pkl file
with open("vectorizer.pkl", "rb") as file:
    test_cv = joblib.load(file)  # loading your vectorizer from the pkl file

# Load your raw data from CSV file
raw = pd.read_csv("test.csv")

# Display the first few rows of the dataframe to confirm it is loaded correctly
print(raw.head())

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit"""

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analysing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")
        
        model_name = ["Logistic Regression Classifier", "Support Vector Classifier", "Random Forest Classifier"]
        # selection = st.sidebar.selectbox("Choose Option", options)
        model_choice = st.selectbox("Select a Classifier Model", model_name)

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = test_cv.transform([news_text]).toarray()
            
            if model_choice == "Logistic Regression Classifier":
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                lg = joblib.load(open(os.path.join("models.pkl"), "rb"))
                prediction = lg.predict(vect_text)
            elif model_choice == "Support Vector Classifier":
                svc_classifier = joblib.load(open(os.path.join("models.pkl"), "rb"))
                prediction = svc_classifier.predict(vect_text)
            elif model_choice == "Random Forest Classifier":
                rf = joblib.load(open(os.path.join("models.pkl"), "rb"))
                prediction = rf.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            if prediction == 1:
                st.success("Text Categorized as: {}".format(prediction))
            elif prediction == 0:
                st.success("Text Categorized as: {}".format(prediction))
            elif prediction == 3:
                st.success("Text Categorized as: {}".format(prediction))
            elif prediction == 4:
                st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()