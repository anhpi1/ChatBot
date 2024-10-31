from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import data_train.library.train_TNN as TNN


number_of_input = 0
number_of_outputs= 0
file_word_list=''
num_words_list= 0
file_input_train=''
number_of_model = 0

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
    if key == "number_of_model":
        if value.isdigit():
            value = int(value)
        number_of_model = value
    if key == "file_word_list":
        file_word_list = value.strip("'")
    if key == "file_input_train":
        file_input_train = value.strip("'")


# tải word-list
with open(file_word_list, 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
tokenizer.word_index = word_index


for i in range(1,number_of_model+1):
    name_mode=i
    file_output_train='data_train/output_train/o{}.ta'.format(name_mode)
    TNN.train_TNN(name_mode ,number_of_input, file_word_list, num_words_list, file_input_train, file_output_train, number_of_outputs)