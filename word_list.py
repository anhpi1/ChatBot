
from tensorflow.keras.preprocessing.text import Tokenizer
import json
max_token=10000

def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()  
    return [sentence.strip() for sentence in sentences]  # xóa ký tự newline

# dữ liệu đầu vào
x = read_sentences_from_file('data_train/x.ta')
print(x)
# tạo word list
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
with open('data_train/word_list.json', 'w') as json_file:
    json.dump(word_index, json_file)

