import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import json
max_token=50

def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()  # Đọc tất cả các dòng
    return [sentence.strip() for sentence in sentences]  # Xóa ký tự newline

# Bước 2: Đọc dữ liệu

x = read_sentences_from_file('x.ta')
# Dữ liệu đầu vào

# Nhãn tương ứng
y = read_sentences_from_file('y.ta')
# Tạo Tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
with open('word_index.json', 'w') as json_file:
    json.dump(word_index, json_file)

