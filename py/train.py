import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import json
max_token=30

# hàm truyền đầu vào
def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]  # xóa ký tự newline

# dữ liệu đầu vào
x = read_sentences_from_file('x.ta')


# dữ liệu đầu ra
y = read_sentences_from_file('y.ta')

# mở word-list
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.word_index = word_index


# mã hóa các câu
X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X, maxlen=max_token)  # pad các chuỗi để có cùng độ dài

# chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.array(y_train, dtype=np.int32)  
y_test = np.array(y_test, dtype=np.int32)    




from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# xây dựng mô hình ANN với dropout
model = Sequential()
model.add(Dense(128, input_shape=(max_token,), activation='relu'))# lớp đầu tiên gồm 128 nút và số lượng input tương ứng
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) # lớp ẩn
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))# lớp đầu ra

# biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# callback cho early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# huấn luyện mô hình với early stopping
model.fit(X_train, y_train, epochs=60, batch_size=2, validation_data=(X_test, y_test), callbacks=[early_stopping])

# đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')
# lưu trọng số và bias vào file
model.save_weights('model_weights.weights.h5')




