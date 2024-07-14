import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk import Text

with open("tfidf_vectorizer.pkl", 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open("nb_classifier.pkl", 'rb') as file:
    nb_classifier = pickle.load(file)

positive_words = ["good", "gerat", "excellent", "wonderful", "amazing"]
negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

encoded = {0: "Business", 1: "Entertainment", 2: "Politics", 3: "Sport", 4: "Technology"}

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ['Home', 'Dataset', 'Classification', 'Visualization', 'Feedback'],
        icons = ['house', 'book', 'envelope', 'box', 'rss'],
        menu_icon = "cast",
    )

# selected = option_menu(
#     menu_title="Main Menu",
#     options=['Home', 'Dataset', 'Classification', 'Visualization', 'Feedback'],
#     icons=['house', 'book', 'envelope', 'box', 'rss'],
#     menu_icon="cast",
#     orientation="horizontal"
# )

##### Home Page ######

if selected == "Home":
    st.title(f"Welcome to Clasifyx 🤖")
    st.markdown("<p style='font-size: 16px;'>Your go-to platform for precise and efficient news headline classification! 📰✨</p>", unsafe_allow_html=True)

    st.markdown(""" ### What is News Headline Classification? """)
    st.markdown("Imagine a system that can automatically sort news headlines into different categories, like Business, Sports, Entertainment, or Politics. This is the power of news headline classification! It helps organize vast amounts of information and makes it easier to find the news you care about. ")



    st.markdown(""" ### Why Choose Clasifyx? 🌟 """)

    st.markdown("""
    ##### 1. Accurate Classification 🔍
    Harness the power of advanced AI technology to categorize news headlines with remarkable accuracy. Our cutting-edge algorithms ensure that each headline is assigned to the most relevant category, saving you time and effort.
    """)

    st.markdown("""
    ##### 2. User-Friendly Interface 🖥️
    Navigate our intuitive platform with ease. Clasifyx offers a seamless user experience, allowing you to quickly classify and retrieve news headlines without any hassle.
    """)

    st.markdown("""
    ##### 3. Real-Time Updates ⚡
    Stay ahead of the curve with real-time updates. Clasifyx constantly scans and classifies incoming news headlines, ensuring you have the latest information at your fingertips.
    """)

    st.markdown("""
    ##### 4. Comprehensive Analytics 📊
    Gain insights into news trends and patterns with our robust analytics tools. Track category distributions, analyze headline trends, and make data-driven decisions effortlessly.
    """)

    st.markdown(""" ### Explore the Power of Classification! """)
    st.markdown("""1. <p style="font-weight: bold"> Dive into Datasets: </p> Explore the data used to train our classification system. See how news headlines are categorized.""", unsafe_allow_html=True)
    st.markdown("""2. <p style="font-weight: bold"> Visualize the News: </p> Interactive charts and graphs reveal trends in word usage and category distribution.""", unsafe_allow_html=True)
    st.markdown("""3. <p style="font-weight: bold"> Classify Your News: </p> Enter a headline and see our system predict its category.""", unsafe_allow_html=True)

##### Home Page End ######

##### Dataset Page ######

if selected == "Dataset":
    st.title("Explore the data used to train our classification system 📊")

    st.markdown("## Dataset")
    df = pd.read_csv("/Users/macbookairm2/Desktop/IIMS/Semister 5/Natural Language Processing and Computer Vision/Assignment/DataSet/Final Data/final_Dataset.csv")
    st.dataframe(df)

    st.markdown("""## DataSet Visualizations""")

    st.markdown("""#### Count Plot""")
    plt.figure(figsize=(8,5))
    colour_types = {
        'Lifestyle': '#8D0801',
        'Sports': '#006400',
        'Automobile': '#FFD700',
        'Business': '#00008B',
        'Politics': '#800080'
    }

    sns.countplot(data=df, x = 'Category', palette=colour_types)
    plt.xlabel("Categories")
    plt.title("Distribution of Categories")
    st.pyplot(plt)

    st.markdown("""#### Box Plot""")
    colour_types = {
        'A': '#FF5733',
        'B': '#2ECC71',
        'C': '#F1C40F',
        'D': '#3498DB', 
        'E': '#9B59B6',
    }

    fig = px.box(df, x='Category', y="Title", color='Category', color_discrete_map=colour_types)
    st.plotly_chart(fig)

    st.markdown("""#### Pie Chart""")
    # category_count = df['Category'].value_counts()
    # plt.figure(figsize=(6,3))
    # plt.pie(category_count, labels=category_count.index, autopct="%1.1f%%", startangle=140)
    # st.pyplot(plt)

    pie = px.pie(df, names='Category')
    st.plotly_chart(pie)








##### Classification Page ######

if selected == "Classification":
    st.title("Classifying News Headlines")

    input_text = st.text_input("Enter News: ")
    input_text = input_text.lower()

    option = st.selectbox(
        "Select Classification Model",
        ("Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine"))

    if st.button("Classify"):

        if input_text.strip() == "":
            st.warning("Please Enter Text to Classify.", icon="⚠️")

        elif len(input_text) <= 60 :
            st.warning("Input length is not sufficient.", icon="⚠️")

        else:
            tfidf_vectorize = tfidf_vectorizer.transform([input_text])
            prediction = nb_classifier.predict(tfidf_vectorize)
            classification = encoded[prediction[0]]
            st.success(f"News Category: {classification}")

            positive_found = any(word in positive_words for word in input_text.split())
            negative_found = any(word in negative_words for word in input_text.split())

            if positive_found and not negative_found:
                st.success("This headline contains positive words.")

            elif negative_found:
                st.error("This headline contains negative words.")

            elif positive_found and negative_found:
                st.info("This headline contains negative and positive words.")
            else:
                st.warning("This headline doesnot contains any negative or positive words.")

            st.markdown("## WordCloud Visualization")

            wordcloud = WordCloud(width=800, height=500, background_color='white').generate(input_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            st.markdown("## Dispersion Plot")

            text = word_tokenize(input_text)
            text1 = Text(text)

            words = [w for w in text if w.isalnum()]

            plt.figure(figsize=(8,5))
            text1.dispersion_plot(words)
            st.pyplot(plt)

            st.markdown("### Visualization")

##### Data Visualization Page ######

if selected == "Visualization":
    st.title("Visualization")
    #
    # word_cloud = WordCloud(width=500, height=400, background_color='white').generate(input_text)
    #
    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
    #
    # ax1.imshow(word_cloud, interpolation='bilinear')
    # ax1.axis('off')
    # st.pyplot(ax1)
    #
    # text = word_tokenize(input_text)
    # text1 = Text(text)
    #
    # target_words = [w for w in text if w.isalnum()]
    #
    # ax2.text1.dispersion_plot(text)
    # st.pyplot(ax2)







if selected == "Feedback":
    st.title("Feedback")

    f_input =  st.text_input("Enter Feedback:")
    if st.button("Send"):
        if f_input.strip() == "":
            st.warning("Enter your Feedback", icon="⚠️")
        else:
            st.success("Thank you for your Feedback.")










