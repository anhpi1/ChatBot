import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import json
max_token=10000

def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()  
    return [sentence.strip() for sentence in sentences]  # xóa ký tự newline



# dữ liệu đầu vào
x = read_sentences_from_file('x.ta')


# dữ liệu đầu ra
y = read_sentences_from_file('y.ta')

# tạo word list
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
with open('word_index.json', 'w') as json_file:
    json.dump(word_index, json_file)

