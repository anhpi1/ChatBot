from tensorflow.keras.preprocessing.text import Tokenizer
import json

num_words_list = 0
file_input_train = ''
file_word_list = ''
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
    if key == "num_words_list":
        if value.isdigit():
            value = int(value)
        num_words_list = value
    if key == "file_input_train":
        file_input_train = value.strip("'")
    if key == "file_word_list":
        file_word_list = value.strip("'")



def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()  
    return [sentence.strip() for sentence in sentences]  # xóa ký tự newline

# dữ liệu đầu vào
x = read_sentences_from_file(file_input_train)

# tạo word list
tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
with open(file_word_list, 'w') as json_file:
    json.dump(word_index, json_file)

