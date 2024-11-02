import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import shutil
# Hàm đọc file
def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# Hàm tạo transformer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Self-attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    # Feed-forward layer
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    return ff_output

def create_model(number_of_outputs, number_of_input, num_words_list):
    # Xây dựng mô hình transformer
    input_layer = Input(shape=(number_of_input,))
    embedding_layer = Embedding(input_dim=num_words_list, output_dim=128, input_length=number_of_input)(input_layer)
    x = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(number_of_outputs, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

def train_TNN(name_mode, number_of_input, file_word_list, num_words_list, file_input_train, file_output_train, number_of_outputs):

    tf.keras.backend.clear_session()

    # Xóa thư mục cache nếu có (thay 'cache_directory' bằng tên thư mục cache)
    cache_dir = 'data_train/library/__pycache__'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    with open(file_word_list, 'r') as json_file:
        word_index = json.load(json_file)

    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index

    # Lấy dữ liệu đào tạo
    input = read_sentences_from_file(file_input_train)
    output = read_sentences_from_file(file_output_train)

    # Mã hóa các câu
    input_sequences = tokenizer.texts_to_sequences(input)
    input_padded = pad_sequences(input_sequences, maxlen=number_of_input)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    input_train, input_test, output_train, output_test = train_test_split(input_padded, output, test_size=0.1)

    # Chuyển đổi output thành array (nếu cần mã hóa nhãn số nguyên)
    output_train = np.array(output_train, dtype=np.int32)
    output_test = np.array(output_test, dtype=np.int32)

    model = create_model(number_of_outputs, number_of_input, num_words_list)

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Điều kiện dừng huấn luyện
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Huấn luyện mô hình
    model.fit(input_train, output_train, epochs=6000, batch_size=2, validation_data=(input_test, output_test), callbacks=[early_stopping],verbose=0)

    # Đánh giá mô hình
    loss, accuracy = model.evaluate(input_test, output_test,verbose=0)
    print(name_mode)
    print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')
    count=0
    while accuracy < 0.9:
        if count >10:
            break
        model.fit(input_train, output_train, epochs=6000, batch_size=2, validation_data=(input_test, output_test), callbacks=[early_stopping], verbose=0)
        
        # Đánh giá mô hình sau mỗi lần huấn luyện
        loss, accuracy = model.evaluate(input_test, output_test, verbose=0)
        count +=1
        print(name_mode)
        print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')

    # Lưu trọng số và bias
    model.save_weights('data_train/weight_model/model_{}.weights.h5'.format(name_mode))

    del model
    # Xóa bộ nhớ cache TensorFlow và giải phóng bộ nhớ

