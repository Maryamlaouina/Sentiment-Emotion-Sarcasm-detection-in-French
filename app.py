# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import pandas as pd 
import numpy as np 
from datetime import datetime
import joblib 
import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model function

# Import necessary libraries for preprocessing
import re
import numpy as np
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import track_utils
# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import math
import json
from keras.utils import custom_object_scope
from transformers import AdamWeightDecay  
from transformers import TFBertModel, BertTokenizer,AdamW
############################sentiments#########################################################
model_s = "models/best_model.pkl"
pipe_s = joblib.load(model_s)
# Fxn
def predict_sentiment(docx):
    results = pipe_s.predict([docx])
    return results[0]
def predict_sentiment2(docx):
    if not isinstance(docx, list):
        docx = [docx]
    results = pipe_s.predict(docx)
    return results
def get_sentiments_proba(docx):
    results = pipe_s.predict_proba([docx])
    return results

sentiments_emoji_dict = {"negative":"➖","positive":"➕"}

############################emotions###########################################################
# Load tokenizer
tokenizer_path = "models/tokenizer.json"
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
    loaded_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

# Load French language model
nlp = spacy.load("fr_core_news_sm")
# Define French stop words
stop_words_french = STOP_WORDS
# Load your pre-trained model
model_path = "models/Emotions.h5"  
pipe_lr = load_model(model_path)
# Your preprocessing functions
def lemmatization_french(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])
    return lemmatized_text

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words_french]
    return " ".join(Text)

def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Your additional preprocessing function
def preprocess_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = Removing_numbers(text)
    text = Removing_punctuations(text)
    text = Removing_urls(text)
    text = lemmatization_french(text)
    return text

# Function to predict emotions
def predict_emotions(docx , tokenizer):
    docx_processed = preprocess_text(docx)
    docx_tokenized = loaded_tokenizer.texts_to_sequences([docx_processed])  # Assuming you have a tokenizer
    docx_padded = pad_sequences(docx_tokenized, maxlen=229)  # Assuming you have a max_sequence_length
    results = pipe_lr.predict(docx_padded)
    return results[0]

#emotions_emoji_dict = {"tristesse": "😔", "colère": "😠",  "surprise": "😮", "amour": "🤗" , "peur": "😨😱", "joie": "🤗"}
emotions_emoji_dict = {"amour": "🤗", "colère": "😠",  "joie": "🤗", "peur": "😨😱" , "surprise": "😮", "tristesse": "😔"}

############################Sarcasm###########################################################
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorflow.keras.optimizers import Adam
from transformers import get_linear_schedule_with_warmup
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from transformers import DistilBertTokenizer, TFDistilBertModel

# Load tokenizer
#tokenizer2 = "models/tokenizer2.json"
#with open(tokenizer2, 'r', encoding='utf-8') as f2:
#   tokenizer_config2 = json.load(f2)
#    loaded_tokenizer2 = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config2)
tokenizer2 = BertTokenizer.from_pretrained('distilbert-base-cased')
# Load French language model
nlp = spacy.load("fr_core_news_sm")
# Define French stop words
stop_words_french = STOP_WORDS
from transformers import TFDistilBertModel

# Define the custom layer
class CustomDistilBertModel(TFDistilBertModel):
    pass

# Streamlit App

tf.keras.utils.get_custom_objects()["AdamW"] = AdamW

# Register the custom optimizer using keras.utils.get_custom_objects()
tf.keras.utils.get_custom_objects()["TFDistilBertModel"] = CustomDistilBertModel

# Load the model without custom_object_scope
#model_sarcasm = tf.keras.models.load_model('models/distillbert.h5')
custom_objects = {
   'TFDistilBertModel': TFDistilBertModel,
    'AdamWeightDecay': AdamWeightDecay  # Utiliser directement AdamWeightDecay
}
model_sarcasm = tf.keras.models.load_model('models/ModeloDistilBert.h5' , custom_objects = custom_objects)


#model = "models/ModeloDistilBert.h5"  

#with custom_object_scope(custom_objects):

#    pipe = load_model(model , dummy_input.any())

def predict_sarcasm(text):
    docx_processed = preprocess_text(text)
    # Use tokenizer.encode to convert text to a list of IDs
    docx_tokenized = tokenizer2.encode(docx_processed, add_special_tokens=True)
    # Assuming dummy_input_2 has the same shape as the second input
    dummy_input_2 = np.zeros((20, ))  # Add an extra dimension for the batch size
    # Pad the sequence for the first input
    docx_padded = pad_sequences([docx_tokenized], maxlen=20, padding='post', truncating='post')  # Adjust maxlen to match the expected input size
    # Assuming your model expects two inputs, use a list for multiple inputs
    dummy_input_2_batch = np.array([dummy_input_2] * docx_padded.shape[0])
      # Wrap the input text in a list to create a batch
    results = model_sarcasm.predict([docx_padded, dummy_input_2_batch])
    #results = model_sarcasm.predict(text)
    return results[0]





sarcasm_dict = {"sarcasm": "✅","non sarcasm": "❌"}


# Main Application 
def main():
    st.title("Classifier : Emotions , Sentiments et Sarcasm")
    menu = ["Emotion", "Sentiments", "Sarcasm"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Emotion":
        add_page_visited_details("Emotion", datetime.now())
        st.subheader("Emotions dans texte")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("écrire ici")
            submit_text = st.form_submit_button(label='Envoyer')
        
        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            prediction_probabilities = predict_emotions(raw_text , loaded_tokenizer) 
            prediction_index = np.argmax(prediction_probabilities)
            prediction_label = list(emotions_emoji_dict.keys())[prediction_index]
            prediction_confidence = prediction_probabilities[prediction_index]
            
            add_prediction_details(raw_text, prediction_label, prediction_confidence, datetime.now())

            with col1:
                st.success("Text original")
                st.write(raw_text)

                st.success("Prediction")

                emoji_icon = emotions_emoji_dict.get(prediction_label)
                st.write("{}:{}".format(prediction_label, emoji_icon))
                st.write("Confidence: {:.2%}".format(prediction_confidence))
                


            with col2:
                st.success("Probabilité de prédiction")
                # Ensure emotions_emoji_dict has 5 keys
                selected_keys = list(emotions_emoji_dict.keys())[:6]

                proba_df = pd.DataFrame({"Probabilité": prediction_probabilities[:6]}, index=selected_keys)
                proba_df.index.name = 'Emotion'
                
                c = alt.Chart(proba_df.reset_index()).mark_bar().encode(
                    x='Emotion',
                    y='Probabilité',
                    color='Emotion'
                )
                st.altair_chart(c, use_container_width=True)
        st.subheader("Emotions dans un fichier :")

        uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

        if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                st.markdown("### Contenu du fichier CSV")
                

                # Create a new DataFrame to store predictions
                predictions_df = pd.DataFrame(columns=["Texte", "Prediction", "Confidence"])
                # Create a list to store data for tabular display
                table_data = []

                for index, row in df.iterrows():
                    prediction_probabilities = predict_emotions(row['Texte'], loaded_tokenizer)
                    prediction_index = np.argmax(prediction_probabilities)
                    prediction_label = list(emotions_emoji_dict.keys())[prediction_index]
                    prediction_confidence = prediction_probabilities[prediction_index]

                    add_prediction_details(row['Texte'], prediction_label, prediction_confidence, datetime.now())

                    emoji_icon = emotions_emoji_dict.get(prediction_label)

                    # Append data for tabular display
                    table_data.append([row['Texte'], f"{prediction_label}: {emoji_icon}", f"{prediction_confidence:.2%}"])

                    # Append the predictions to the new DataFrame
                    predictions_df = predictions_df.append({
                        "Texte": row['Texte'],
                        "Prediction": prediction_label,
                        "Confidence": prediction_confidence
                    }, ignore_index=True)

                # Display the predictions in a table
                st.table(pd.DataFrame(table_data, columns=["Texte", "Prediction", "Confidence"]))

                # Provide an option to download the predictions as a new CSV file
                if st.button("Télécharger les prédictions CSV"):
                    predictions_file = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les prédictions",
                        data=predictions_file,
                        file_name="predictions.csv",
                        key="predictions_csv"
                    )
               
    if choice == "Sentiments":
        add_page_visited_details("Sentiments", datetime.now())
        st.subheader("Sentiments dans texte")

        with st.form(key='sentiment_clf_form'):
            raw_text2 = st.text_area("écrire ici")
            submit_text2 = st.form_submit_button(label='Envoyer')
        # ...

        if submit_text2:
            col1, col2 = st.columns(2)



            prediction = predict_sentiment(raw_text2)
            pred_index = np.argmax(prediction)
            pred_label = list(sentiments_emoji_dict.keys())[prediction]
            probability = get_sentiments_proba(raw_text2)

            # Apply Fxn Hereé
            
            add_prediction_details(raw_text2,prediction,np.max(probability),datetime.now())


            with col1:
                st.success("Text original")
                st.write(raw_text2)

                st.success("Prediction")

                emoji_icon2 = sentiments_emoji_dict.get(pred_label)
                st.write("{}:{}".format(pred_label, emoji_icon2))
                st.write("Confidence: {:.2%}".format(math.ceil(probability[0, 0])))

            with col2:
                st.success("Probabilité de prédiction")
                # Ensure emotions_emoji_dict has 5 keys
                selected_keys2 = list(sentiments_emoji_dict.keys())[:2]

                proba_df2 = pd.DataFrame({"Probabilité": probability[:2].flatten()}, index=selected_keys2)
                proba_df2.index.name = 'Sentiments'
                
                c = alt.Chart(proba_df2.reset_index()).mark_bar().encode(
                    x='Sentiments',
                    y='Probabilité',
                    color='Sentiments'
                )
                st.altair_chart(c, use_container_width=True)

        st.subheader("Sentiments dans un fichier :")

        uploaded_se = st.file_uploader("Uploader un fichier CSV", type=["csv"])

        if uploaded_se is not None:
                df_se = pd.read_csv(uploaded_se, encoding='ISO-8859-1')
                st.markdown("### Contenu du fichier CSV")
                

                # Create a new DataFrame to store predictions
                predictions_se = pd.DataFrame(columns=["Texte", "Prediction"])
                # Create a list to store data for tabular display
                table_se = []
                # Convert all 'Texte' column values to strings to handle any non-string data
                df_se['Texte'] = df_se['Texte'].astype(str)
                # Predict sentiment for all rows in 'Texte' column
                
                for index2, row2 in df_se.iterrows():
                    prediction_probabilities2 = predict_sentiment2([row2['Texte']])
                    prediction_class2 =  prediction_probabilities2[0]
                    prediction_label2 = list(sentiments_emoji_dict.keys())[prediction_class2]
                    confidence2 = get_sentiments_proba(row2['Texte'])

                    add_prediction_details(row2['Texte'], prediction_label2, confidence2, datetime.now())

                    senti_icon = sentiments_emoji_dict.get(prediction_label2)

                    # Append data for tabular display
                    table_se.append([row2['Texte'], f"{prediction_label2}: {senti_icon}"])

                    # Append the predictions to the new DataFramejj
                    predictions_se = predictions_se.append({
                        "Texte": row2['Texte'],
                        "Prediction": prediction_label2,
                        
                    }, ignore_index=True)

                # Display the predictions in a table
                st.table(pd.DataFrame(table_se, columns=["Texte", "Prediction"]))

                # Provide an option to download the predictions as a new CSV file
                if st.button("Télécharger les prédictions CSV"):
                    predictions_file2 = predictions_se.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les prédictions",
                        data=predictions_file2,
                        file_name="predictions_sentiments.csv",
                        key="predictions_sentiments_csv"
                    )
    if choice == "Sarcasm":
        add_page_visited_details("Sarcasm", datetime.now())
        st.subheader("Sarcasm dans texte")

        with st.form(key='sarcasm_clf_form'):
            raw_text3 = st.text_area("écrire ici")
            submit_text3 = st.form_submit_button(label='Envoyer')
        
        if submit_text3:
            col1, col2 = st.columns(2)
            
            threshold = 0.5
            prediction3 = predict_sarcasm(raw_text3) 
            label3 = 'sarcasm' if prediction3[0] > threshold else 'non sarcasm'
            confidence3_sarcasm = prediction3[0] if label3 == 'sarcasm' else 1 - prediction3[0]
            #confidence3_non_sarcasm = 1 - confidence3_sarcasm if label3 == 'sarcasm' else confidence3_sarcasm

            
            
            add_prediction_details(raw_text3, label3, confidence3_sarcasm, datetime.now())

            with col1:
                st.success("Text original")
                st.write(raw_text3)

                st.success("Prediction")

                emoji_icon3 = sarcasm_dict.get(label3)
                st.write("{}:{}".format(label3, emoji_icon3))
                st.write("Confidence: {:.2%}".format(confidence3_sarcasm))
               

    
            
            with col2:
                st.success("Probabilité de prédiction")
                confidence_sarcasm = prediction3[0] 
                confidence_non_sarcasm = 1 - confidence_sarcasm 


# Create a DataFrame with both probabilities
                proba_df3 = pd.DataFrame({
                       "Sarcasm": ["sarcasm", "non sarcasm"],
                       "Confidence": [confidence_sarcasm, confidence_non_sarcasm]  # Confidence for sarcasm and non sarcasm
                  })

                proba_df3.index.name = 'Sarcasm'

# Create a stacked bar chart to show both probabilities and confidence
                c = alt.Chart(proba_df3).mark_bar().encode(
                x='Sarcasm',
                y=('Confidence'),
                color='Sarcasm')
                
                st.altair_chart(c, use_container_width=True)
        st.subheader("Sarcasm dans un fichier :")

        uploaded_sa = st.file_uploader("Uploader un fichier CSV", type=["csv"])

        if uploaded_sa is not None:
                df_sa = pd.read_csv(uploaded_sa, encoding='ISO-8859-1')
                st.markdown("### Contenu du fichier CSV")
                

                # Create a new DataFrame to store predictions
                predictions_sa = pd.DataFrame(columns=["Texte", "Prediction", "Confidence"])
                # Create a list to store data for tabular display
                table_sa = []
                threshold2 = 0.5
                for index3, row3 in df_sa.iterrows():
                    prediction_probabilities3 = predict_sarcasm(row3['Texte'])
                    prediction_label3 = 'sarcasm' if prediction_probabilities3[0] > threshold2 else 'non sarcasm'
                    prediction_confidence3 = prediction_probabilities3[0] if prediction_label3 == 'sarcasm' else 1 - prediction_probabilities3[0]

                    add_prediction_details(row3['Texte'], prediction_label3, prediction_confidence3, datetime.now())

                    emoji_icon33 = sarcasm_dict.get(prediction_label3)

                    # Append data for tabular display
                    table_sa.append([row3['Texte'], f"{prediction_label3}: {emoji_icon33}", f"{prediction_confidence3:.2%}"])

                    # Append the predictions to the new DataFrame
                    predictions_sa = predictions_sa.append({
                        "Texte": row3['Texte'],
                        "Prediction": prediction_label3,
                        "Confidence": prediction_confidence3
                    }, ignore_index=True)

                # Display the predictions in a table
                st.table(pd.DataFrame(table_sa, columns=["Texte", "Prediction", "Confidence"]))

                # Provide an option to download the predictions as a new CSV file
                if st.button("Télécharger les prédictions CSV"):
                    predictions_file3 = predictions_sa.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les prédictions",
                        data=predictions_file3,
                        file_name="sarcasm_predictions.csv",
                        key="sarcasm_csv"
                    )



if __name__ == '__main__':
    main()
