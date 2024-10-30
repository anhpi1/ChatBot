from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import data_train.library.train_TNN as TNN


number_of_input = 30
file_word_list='data_train/word_list.json'
num_words_list=10000
file_input_train='data_train/input_train/content_question.ta'

# tải word-list
with open(file_word_list, 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
tokenizer.word_index = word_index


for i in range(1,7):
    name_mode=i
    file_output_train='data_train/output_train/o{}.ta'.format(name_mode)
    number_of_outputs=10
    TNN.train_TNN(name_mode ,number_of_input, file_word_list, num_words_list, file_input_train, file_output_train, number_of_outputs)