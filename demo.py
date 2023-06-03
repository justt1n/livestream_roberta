import transformers
import csv
import datetime
import numpy as np
import sys
import tensorflow as tf
import re, string
import emoji

from transformers import BertTokenizer, BertForMaskedLM
from tensorflow import keras
from chat_downloader import ChatDownloader


def LoadModel(model_id):
  print("Loading model...")
  try:
    if model_id == 1:
      model = keras.models.load_model('bert.h5', custom_objects={"TFBertModel": transformers.TFBertModel})
    else:
      model = keras.models.load_model('roberta.h5', custom_objects={"TFRobertaModel": transformers.TFRobertaModel})
    print("Model loaded!")
    return model
  except:
    print("Can not load model!\nQuiting...")
    sys.exit(1)

def LoadTokenizer(model_id):
  print("Loading tokenizer...")
  try:
    if model_id == 1:
      tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    else:
      tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
    print("Tokenizer Loaded!")
    return tokenizer
  except:
    sys.exit(2)

# def strip_emoji(text):
#     return re.sub(emoji.get_emoji_regexp(), r"", text) 

# def clean_hashtags(tweet):
#     new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) 
#     new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) 
#     return new_tweet2

# def filter_chars(a):
#     sent = []
#     for word in a.split(' '):
#         if ('$' in word) | ('&' in word):
#             sent.append('')
#         else:
#             sent.append(word)
#     return ' '.join(sent)

# def remove_mult_spaces(text): 
#     return re.sub("\s\s+" , " ", text)

# def strip_all_entities(text): 
#     text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
#     text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
#     text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
#     banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
#     table = str.maketrans('', '', banned_list)
#     text = text.translate(table)
#     return text

# def cleanText(text):
#   print("Clean text...")
#   return remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(text)))))

def MessagePredict(str, model, tokenizer):
  MAX_LEN = 128
  # Tokenize the sentence using the BERT tokenizer
  tokens = tokenizer.tokenize(str)

  # Add special tokens to the start and end of the sequence
  tokens = ['[CLS]'] + tokens + ['[SEP]']

  # Convert tokens to ids using the BERT tokenizer
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # Pad the input sequence to the maximum length
  input_ids = input_ids + [0] * (MAX_LEN - len(input_ids))

  # Create the attention mask for the input sequence
  attention_masks = [int(id > 0) for id in input_ids]

  # Reshape the input and mask as required by the model
  input_ids = np.array(input_ids).reshape((1, MAX_LEN))
  attention_masks = np.array(attention_masks).reshape((1, MAX_LEN))

  # Make a prediction using the model
  prediction = model.predict([input_ids, attention_masks])[0]

  # The predicted class is the index of the highest value in the prediction vector
  predicted_class = np.argmax(prediction)


  if predicted_class == 0:
    return "Negative"
  elif predicted_class == 1:
    return "Neutral"
  else:
    return "Positive"
  
model_id = int(input("Chose model:\n1: BERT\n2: RoBERTa\nModel: "))

model = LoadModel(model_id)
tokenizer = LoadTokenizer(model_id)



url = str(input("Nhap url: "))

chat = ChatDownloader().get_chat(url)  
now = datetime.datetime.now()
time_str = now.strftime("%H%M%S")
filename = f"livestream_chat_{time_str}.csv"
header = ["time", "user_id", "username", "content", "sentiment"]

with open(filename, mode='w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(header)

  for message in chat:
    data_time = message['timestamp']
    data_time = datetime.datetime.fromtimestamp(data_time / 1000000)
    data_user_id = message['author']['id']
    data_username = message['author']['name']
    data_content = message['message']
    data_sentiment = MessagePredict(message['message'], model,tokenizer)
    if data_sentiment == "Negative":
      print("Found Negative Comment")
    writer.writerow([data_time, data_user_id, data_username, data_content, data_sentiment])
  
