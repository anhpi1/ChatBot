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

# mowr file word_list
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.word_index = word_index

# Mã hóa các câu
X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X, maxlen=max_token)  # Pad các chuỗi để có cùng độ dài

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi y thành numpy array và đảm bảo là kiểu số nguyên
y_train = np.array(y_train, dtype=np.int32)  # Đảm bảo là số nguyên
y_test = np.array(y_test, dtype=np.int32)    # Đảm bảo là số nguyên

# Tạo mô hình
model = Sequential()
model.add(Dense(128, input_shape=(max_token,), activation='relu'))  # Kích thước đầu vào là độ dài của các câu đã pad
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Sử dụng số lớp đầu ra phù hợp

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=20, batch_size=2, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')
# Lưu trọng số và bias vào file với đúng định dạng tên file
model.save_weights('model_weights.weights.h5')




