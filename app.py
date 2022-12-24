import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
pipe = pickle.load(open('pipe.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()

def basic_clean(x) :
    words = word_tokenize(x.lower())
    final = []
    
    
    for word in words :     
        if word.isalnum():
            final.append(word)
    

    words = final[:]
    final= []
    
    for i in words :
        if i not in stop_words and i not in string.punctuation :
            final.append(i)
    words = final[:]
    final= []
    
    for i in words :
        final.append(ps.stem(i))
    words = final[:]
    final= []
    return ' '.join(words)


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = basic_clean(input_sms)
    print(transformed_sms)
    # 2. vectorize
    vector_input = pipe.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    print(result)
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
