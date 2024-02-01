import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import docx2txt
import PyPDF2

final_vectorizer = pickle.load(open('final_vectorizer.pkl', 'rb'))
final_model = pickle.load(open('final_model.pkl', 'rb'))

def fake_news_prediction(news):
    port_stem = PorterStemmer()
    con = re.sub('[^a-zA-Z]', ' ', news)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    
    input_data = [con]
    vectorized_data = final_vectorizer.transform(input_data)
    
    prediction = final_model.predict(vectorized_data)
    prediction_proba = final_model.predict_proba(vectorized_data)[0]  
    return prediction, prediction_proba

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    article_text = ' '.join([paragraph.text for paragraph in soup.find_all('p')])
    
    return article_text

def read_docx(file):
    text = docx2txt.process(file)
    return text

def read_pdf(file):
    with BytesIO(file.read()) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def main():
    st.image("q.png", use_column_width=True, caption="Let's Check It...")

    option = st.sidebar.radio("Select an option:", ("Enter News", "Enter URL", "Upload Word/PDF"))

    if option == "Enter News":
        st.subheader("Enter News Paragraph:")
        news_input = st.text_area("Type or paste the news paragraph here:")

        if st.button("Detect Fake News"):
            if news_input:
                prediction_result, prediction_proba = fake_news_prediction(news_input)

                st.subheader("Prediction Result:")
                if prediction_result == [0]:
                    st.success('Reliable')
                    st.success('Since it is reliable you can proceed with it')
                else:
                    st.error('Unreliable')

                    if prediction_proba[1] > 0.5:  
                        st.warning('The news is potentially unreliable.')
                        st.warning('Suggestions: Consider fact-checking, informing users about potential misinformation, etc.')
                    else:
                        st.success('The news is predicted to be reliable. No further actions are required.')

                st.subheader("Prediction Probability:")
                labels = ['Reliable', 'Unreliable']
                colors = ['#66ff66', '#ff6666']

                fig, ax = plt.subplots()
                ax.pie(prediction_proba, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  
                st.pyplot(fig)

    elif option == "Enter URL":
        st.subheader("Enter News URL:")
        url_input = st.text_input("Type or paste the URL of the news article:")

        if st.button("Scrape and Detect Fake News"):
            if url_input:
                article_content = scrape_article(url_input)

                prediction_result, prediction_proba = fake_news_prediction(article_content)

                st.subheader("Prediction Result:")
                if prediction_result == [0]:
                    st.success('Reliable')
                else:
                    st.error('Unreliable')

                    if prediction_proba[1] > 0.5:  
                        unreliable_urls = pd.DataFrame({'Unreliable_URL': [url_input]})
                        unreliable_urls.to_csv('unreliable_urls.csv', mode='a', header=False, index=False)

                        st.warning(f'This webpage ({url_input}) will be reported.')

                    st.warning('Suggestions: Consider fact-checking, informing users about potential misinformation, etc.')

                st.subheader("Prediction Probability:")
                labels = ['Reliable', 'Unreliable']
                colors = ['#66ff66', '#ff6666']

                fig, ax = plt.subplots()
                ax.pie(prediction_proba, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal') 
                st.pyplot(fig)

    elif option == "Upload Word/PDF":
        st.subheader("Upload Word or PDF File:")
        uploaded_file = st.file_uploader("Choose a file", type=["docx", "pdf"])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'docx':
                news_input = read_docx(uploaded_file)
            elif file_extension == 'pdf':
                news_input = read_pdf(uploaded_file)

            prediction_result, prediction_proba = fake_news_prediction(news_input)

            st.subheader("Prediction Result:")
            if prediction_result == [0]:
                st.success('Reliable')
                st.success('Since it is reliable you can proceed with it')
            else:
                st.error('Unreliable')

                if prediction_proba[1] > 0.5:  
                    st.warning('The news is potentially unreliable.')
                    st.warning('Suggestions: Consider fact-checking, informing users about potential misinformation, etc.')
                else:
                    st.success('The news is predicted to be reliable. No further actions are required.')

            st.subheader("Prediction Probability:")
            labels = ['Reliable', 'Unreliable']
            colors = ['#66ff66', '#ff6666']

            fig, ax = plt.subplots()
            ax.pie(prediction_proba, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  
            st.pyplot(fig)

if __name__ == "__main__":
    main()
