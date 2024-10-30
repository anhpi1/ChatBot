import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import data_train.library.train_TNN as TNN

number_of_input = 30
file_word_list = 'data_train/word_list.json'
num_words_list = 10000
number_of_outputs = 10

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
