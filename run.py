import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
max_token=30
# mở word-list
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.word_index = word_index

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# xây dựng mô hình ANN với dropout
model = Sequential()
model.add(Dense(128, input_shape=(max_token,), activation='relu'))# lớp đầu tiên gồm 128 nút và số lượng input tương ứng
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) # lớp ẩn
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))# lớp đầu ra
# tải mô hình đã được đào tạo từ trước
model.load_weights('model_weights.weights.h5')

input_sentence = "Tigers are known for their stealth and can approach prey silently before pouncing."

 
sequence = tokenizer.texts_to_sequences([input_sentence])
padded_sequence = pad_sequences(sequence,maxlen=max_token)


predictions = model.predict(padded_sequence)
print(predictions)

predicted_class = np.argmax(predictions, axis=1)  
print(predicted_class)





