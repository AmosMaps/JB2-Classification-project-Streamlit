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
    
# Load your raw data
raw = pd.read_csv("train.csv")

# Function to display information page
def show_information():
    st.info("In today's digital world, people read a lot more news than ever before, creating a huge amount of information for readers to sort through. To help readers find the news they care about, we aim to build a model that organizes news articles into predefined categories based on their subject matter. This will help in systematically managing and retrieving content, making it more accessible and relevant to users.")
    st.markdown("Using new technologies in machine learning and natural language processing (NLP), this project will create robust models that can automatically sort news articles into the right categories. This will lead to improved content categorization, operational efficiency, and enhanced user experience, which are crucial for a news outlet. These models will later be turned into an app for news outlets and readers to use to their advantage.")
    
# Function to display prediction page
def show_prediction():
    st.info("Prediction with Machine Learning Models")
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
    st.markdown(
        """
        <style>
        .header {
            margin-bottom: 0.5rem;
        }
        .subheader {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="header">What we do</h2>', unsafe_allow_html=True)
    st.write(
        """
        As Data Science consultants specializing in app development, we craft bespoke solutions tailored precisely to your preferences and needs. From translating your ideas into robust software to enhancing functionality with cutting-edge machine learning capabilities, we strive to make your vision a reality. Our goal is to not only meet but exceed your expectations, delivering intuitive applications that simplify and enrich your daily operations. Let us transform your aspirations into powerful tools that elevate efficiency and innovation in your endeavors.
        """
    )
    
    st.markdown('<h2 class="subheader">Meet our team</h2>', unsafe_allow_html=True)
    st.write(
        """
        Amos Maponya - Software Engineer

        Siphosethu Rululu - Data Scientist

        Lebogang Malata - Project Manager

        Judith Kabongo - GitHub Manager

        Josephine Ndukwani - Data Scientist

        Tselane Moeti - Data Scientist
        """
    )
    
    with st.container():
        st.write("---")
        st.header(":mailbox_with_mail: Get In Touch With Us!")
        st.write("##")

        contact_form = """
        <form action="https://formsubmit.co/amosphashe@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message"></textarea>
            <button type="submit">Send</button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
    
    with st.container():
        st.write("##")
        st.header(":round_pushpin: Find us here")
        st.write("Suite 500, Pinnacle Building, 123 Main Street, Cape Town, 8000")

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
