from logging import PlaceHolder
import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


tfidf = pickle.load(open('vectorize.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))



def txt_transform(txt):
    txt = txt.lower() #lowercase text
    txt =  nltk.word_tokenize(txt) #splitting text

    ps = PorterStemmer()
    punct = string.punctuation
    final = []


    for char in txt:
        #if character is an alpha numeric or
        #special char then append it to this array
        if char.isalnum() and char not in stopwords.words('english') and char not in punct:
            final.append(ps.stem(char))

    return " ".join(final)



st.title("Simple Email / Text Spam Classifier")

placeholder = st.empty()
input_txt = placeholder.text_area("Enter text to classify as Spam or Not spam.")




if st.button("Predict Message") and input_txt is not "":
    
    transformed_text = txt_transform(input_txt)

    vector_input = tfidf.transform([transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Mostly Spam :/")
    else:
        st.header("Mostly NOT Spam ;)")

else:
    st.header("Please enter text to get required results.\nNo text added to make prediction.\n")

if st.button("Clear input"):
    input_txt = placeholder.text_area('Enter text to classify as Spam or Not spam.',value='',key=1)   
