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

def train_TNN(name_mode, number_of_input, file_word_list, num_words_list, file_input_train, file_output_train, number_of_outputs,number_of_copies_model):
    # Tải tham số
    with open("parameter.ta", "r") as file:
        lines = file.readlines()
    for line in lines:
        if not line.strip():  # Bỏ qua dòng trống
            continue
        key, value = line.split(" = ")
        key, value = key.strip(), value.strip()
        if key == "weight_model":
            weight_model = value.strip("'")
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
    for i in range(0,number_of_copies_model):
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
        print("mode:{}{}".format(name_mode,i))
        print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')
        
        # Lưu trọng số và bias
        model.save_weights(weight_model.format(name_mode,i))

    del model
    # Xóa bộ nhớ cache TensorFlow và giải phóng bộ nhớ

def update_weights_on_incorrect_prediction(model, incorrect_sentence_padded, correct_label):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    count=0
    correct_label = np.array([correct_label])
    while True:
        with tf.GradientTape() as tape:
            logits = model(incorrect_sentence_padded, training=True)
            loss_value = loss_fn(correct_label, logits)
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        count += 1
        if (loss_value < 0.0001) or (count > 1000):
            break
    # print("correct lable:{}".format(correct_label))
    # print(f"Trọng số đã được cập nhật với loss: {loss_value.numpy():.4f}")

def update_weights_models(name_models, input, correct_label):
    # Khởi tạo tham số
    number_of_input = 0
    file_word_list = ''
    num_words_list = 0
    number_of_outputs = 0
    number_of_copies_model = 0

    # Tải tham số
    with open("parameter.ta", "r") as file:
        lines = file.readlines()
    for line in lines:
        if not line.strip():  # Bỏ qua dòng trống
            continue
        key, value = line.split(" = ")
        key, value = key.strip(), value.strip()
        if key == "number_of_input" and value.isdigit():
            number_of_input = int(value)
        elif key == "num_words_list" and value.isdigit():
            num_words_list = int(value)
        elif key == "number_of_copies_model" and value.isdigit():
            number_of_copies_model = int(value)
        elif key == "file_word_list":
            file_word_list = value.strip("'")
        elif key == "output_train":
            output_train = value.strip("'")
        elif key == "weight_model":
            weight_model = value.strip("'")
    # Tải word index
    try:
        with open(file_word_list, 'r') as json_file:
            word_index = json.load(json_file)
    except FileNotFoundError:
        print(f"Không tìm thấy tệp danh sách từ {file_word_list}.")
        return None

    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index

    # Đảm bảo input là một danh sách các câu
    if isinstance(input, str):
        input = [input]

    input_sequences = tokenizer.texts_to_sequences(input)
    incorrect_sentence_padded = pad_sequences(input_sequences, maxlen=number_of_input)

    # Đảm bảo correct_label có định dạng (batch_size,)
    correct_label = np.array([correct_label])

    # Tải các mô hình với trọng số tương ứng và cập nhật trọng số
    for i in range(number_of_copies_model):
        file_output_train = output_train.format(name_models)
        try:
            with open(file_output_train, "r") as file:
                numbers = file.readlines()
            number_of_outputs = max(int(number.strip()) for number in numbers) + 1
        except FileNotFoundError:
            print(f"Không tìm thấy tệp đầu ra train {file_output_train}.")
            continue

        new_model = create_model(number_of_outputs, number_of_input, num_words_list)
        new_model.load_weights(weight_model.format(name_models,i))
        # print("input:{}".format(input))
        # print("{}{}".format(name_models,i))

        # Gọi hàm update_weights_on_incorrect_prediction với đầu vào đã điều chỉnh
        update_weights_on_incorrect_prediction(new_model, incorrect_sentence_padded, correct_label)
        new_model.save_weights(weight_model.format(name_models,i))    
        