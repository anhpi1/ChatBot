import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
max_token=50
# Tải từ điển từ tệp JSON
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.word_index = word_index
# Tạo mô hình
model = Sequential()
model.add(Dense(128, input_shape=(max_token,), activation='relu'))  # Kích thước đầu vào là độ dài của các câu đã pad
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Sử dụng số lớp đầu ra phù hợp
# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Tải trọng số đã lưu vào mô hình
model.load_weights('model_weights.weights.h5')
# Câu cần dự đoán
input_sentence = "Tigers are the largest species of the cat family."

# Mã hóa câu
sequence = tokenizer.texts_to_sequences([input_sentence])
padded_sequence = pad_sequences(sequence,maxlen=max_token)

# Dự đoán
predictions = model.predict(padded_sequence)
print(predictions)
# In kết quả dự đoán
predicted_class = np.argmax(predictions, axis=1)  # Lấy chỉ số của lớp có xác suất cao nhất
print(predicted_class)





