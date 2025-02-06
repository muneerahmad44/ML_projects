
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import sklearn
import pickle

df=pd.read_csv('/content/drive/MyDrive/emails.csv')

df.head()

df.info()

"""

1.   Data Preprocessing and Exploration

"""

df.isnull().sum()#no null values

df['spam'].value_counts()

sns.countplot(x=df['spam'],data=df)

#1. lowercasing the text column
df['text']=df['text'].str.lower()

import re
def remove_tags(text):
  pattern = re.compile('<.*?>')
  return pattern.sub(r'',text)

df['text']=df['text'].apply(remove_tags)

df['text']

#removing urls from the text column
def remove_url(text):
  pattern=re.compile(r'https?://\S+|www\.\S+')
  return pattern.sub(r'',text)

df['text']=df['text'].apply(remove_url)

import re

def remove_punc(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['text'] = df['text'].apply(remove_punc)

df['text']

slang_dict = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A**",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing"
}

#replace chat slang words
def replace_slang_words(text):
    new_text = []
    for w in text.split():
        if w.upper() in slang_dict:
            new_text.append(slang_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)  # Join with spaces

df['text']=df['text'].apply(replace_slang_words)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Define the stop word removal function using list comprehension
def stop_word_removal(text):
    return " ".join([word for word in text.split() if word not in stop_words])

# Apply the function to your DataFrame
df['text'] = df['text'].apply(stop_word_removal)

!pip install emoji

import emoji
def remove_emoji(text):
  return emoji.demojize(text)
df['text']=df['text'].apply(remove_emoji)

import spacy

# Import necessary libraries
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")  # You might need to download this model first: python -m spacy download en_core_web_sm

# Process the text data using spaCy
docs = list(nlp.pipe(df['text']))

# Extract tokens and store them as strings in a new column
df['tokens4'] = [[token.text for token in doc] for doc in docs]

with open('nlp_object_tokenization.pkl','wb') as f:
  pickle.dump(nlp,f)

df['tokens4']

'''from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
stemmer = PorterStemmer()
def apply_stemmer(text):

# Apply stemming to each word
  return  " ".join([stemmer.stem(word) for word in text])

df['tokens4'].apply(apply_stemmer)'''

print(df['tokens4'].apply(type).value_counts())

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag

# Download necessary NLTK resources if not already downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt') # This line was added to handle sentence tokenization within pos_tag
nltk.download('averaged_perceptron_tagger_eng') # This line was added to download the missing resource

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def apply_lemmatizer(tokens):
    """Apply lemmatization to a list of tokens."""
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]

# Apply the lemmatization function to the 'tokens4' column
df['lemmatized_text'] = df['tokens4'].apply(apply_lemmatizer)

with open('lemmatizer.pkl','wb') as f:
  pickle.dump(lemmatizer,f)

df['lemmatized_text']

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Create a dictionary (vocabulary) from the spaCy tokens
word_index = {}
for sentence in df['lemmatized_text']:  # Loop through each row in the lemmatized text column
    for token in sentence:
        if token not in word_index:
            word_index[token] = len(word_index) + 1  # Assign a unique integer to each token

# Step 2: Convert the tokens in the 'lemmatized_text' column to integer sequences
def tokens_to_sequence(tokens):
    return [word_index.get(token, 0) for token in tokens]  # Default to 0 for unknown words

# Convert the entire 'lemmatized_text' column to sequences of integers
sequences = df['lemmatized_text'].apply(tokens_to_sequence)

# Step 3: Pad sequences to ensure uniform length (e.g., 100 tokens per sequence)
X = pad_sequences(sequences, padding='post', maxlen=100)  # Adjust maxlen as necessary

with open('word_index.pkl','wb') as f:
  pickle.dump(word_index,f)

from sklearn.model_selection import train_test_split

# Step 1: Split the data into training and validation sets

X_train, X_test, y_train, y_test=train_test_split(X, df['spam'], test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val =train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 2: Define the model (as explained earlier)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=300),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['precision'])

# Step 4: Train the model on the training data and evaluate on the validation set
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save("spam_email.keras")

model.evaluate(X_test,y_test)

y_prediction=model.predict(X_test)
y_pred=(y_prediction > 0.5).astype(int)

from sklearn.metrics import classification_report



# Generate classification report
report = classification_report(y_test, y_pred)

# Print the report
print(report)

!pip install streamlit

#%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import emoji
import pickle
#import spacy
#from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download necessary NLTK data (only if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


model=load_model("/content/spam_email.keras")
# Load the saved model (replace 'your_model.h5' with the actual filename)
with open("word_index.pkl", "rb") as f:
        word_index1=pickle.load(f)
with open('nlp_object_tokenization.pkl','rb') as f:
        nlp=pickle.load(f)

'''
@st.cache_resource
def load_spam_model():


@st.cache_resource
def load_word_index():
    with open("word_index.pkl", "rb") as f:
        return pickle.load(f)

'''

# Preprocessing functions (same as in your notebook)
def remove_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_punc(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

slang_dict = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A**",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing"
}
 # Your slang dictionary from the notebook


#replace chat slang words
def replace_slang_words(text):
    new_text = []
    for w in text.split():
        if w.upper() in slang_dict:
            new_text.append(slang_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)  # Join with spaces

 # ... your slang replacement function ...} # Your slang replacement function from the notebook

stop_words = set(stopwords.words('english'))
def stop_word_removal(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_emoji(text):
  return emoji.demojize(text)

def apply_lemmatizer(tokens):

    with open('lemmatizer.pkl','rb') as f:
      lemmatizer=pickle.load(f)
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]



def preprocess_text(text):
    text = text.lower()
    text = remove_tags(text)
    text = remove_url(text)
    text = remove_punc(text)
    text = replace_slang_words(text)
    text = stop_word_removal(text)
    text = remove_emoji(text)

    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmatized_tokens = apply_lemmatizer(tokens)
    return lemmatized_tokens


def predict(text):
    #word_index1=load_word_index()
    processed_text = preprocess_text(text)
    sequence = [word_index1.get(token, 0) for token in processed_text]
    padded_sequence = pad_sequences([sequence], padding='post', maxlen=100)
    #model=load_spam_model()  # Adjust maxlen if needed
    prediction = model.predict(padded_sequence)[0][0]
    return prediction

# Streamlit app
st.title("Spam Email Classifier")

user_input = st.text_area("Enter email text:", "")

if st.button("Predict"):
    if user_input:
        prediction = predict(user_input)
        if prediction > 0.5:
            st.error("Spam")
        else:
            st.success("Not Spam")
    else:
        st.warning("Please enter some text")

!killall ngrok

pip install pyngrok

from pyngrok import ngrok

# Kill any existing ngrok processes


# Set your Ngrok authtoken (replace 'your_auth_token' with the copied token)
ngrok.set_auth_token('2bOYa4qRr7z4K3HOLkfhBIJgRPB_5Th4f73XnfSqFC8oG4o8S')

# Set up the Ngrok tunnel to the Streamlit app
# The port number should be included in the 'addr' argument
public_url = ngrok.connect(addr='http://localhost:8501')
print(f"Streamlit app is live at: {public_url}")

!streamlit run app.py &

