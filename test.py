import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import data_train.library.train_TNN as TNN

number_of_input = 0
file_word_list = ''
num_words_list = 0
number_of_outputs = 0

# tải tham số
with open("parameter.ta", "r") as file:
    lines = file.readlines()
for line in lines:
    # Bỏ qua các dòng trống
    if not line.strip():
        continue
    # Tách dòng thành key và value
    key, value = line.split(" = ")
    key = key.strip()
    value = value.strip()
    # Kiểm tra nếu value là số nguyên trước khi chuyển đổi
    if key == "number_of_input":
        if value.isdigit():
            value = int(value)
        number_of_input = value
    if key == "number_of_outputs":
        if value.isdigit():
            value = int(value)
        number_of_outputs = value
    if key == "num_words_list":
        if value.isdigit():
            value = int(value)
        num_words_list = value
    if key == "file_word_list":
        file_word_list = value.strip("'")

# Tải word-list
with open(file_word_list, 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
tokenizer.word_index = word_index

models = []

# Tạo và tải các mô hình từ các trọng số
for name_mode in range(1, 7):
    new_model = TNN.create_model(number_of_outputs, number_of_input, num_words_list)
    new_model.load_weights('data_train/weight_model/model_{}.weights.h5'.format(name_mode))
    models.append(new_model)  # Thêm mô hình mới vào danh sách

# Câu cần dự đoán
input_sentence = "i love my cat, my cat is so dump and i like this"

# Mã hóa câu
sequence = tokenizer.texts_to_sequences([input_sentence])
padded_sequence = pad_sequences(sequence, maxlen=number_of_input)

# Chuyển đổi padded_sequence thành numpy array để dự đoán
padded_sequence = np.array(padded_sequence)

# Dự đoán cho từng mô hình
for index, model in enumerate(models):
    # Dự đoán
    predictions = model.predict(padded_sequence, verbose=0)  # Tắt chế độ verbose
    print(f'Predicted Probabilities for Model {index + 1}: {predictions}')

    # In kết quả dự đoán
    predicted_class = np.argmax(predictions, axis=1)  # Lấy chỉ số của lớp có xác suất cao nhất
    print(f'Predicted Class for Model {index + 1}: {predicted_class}')
