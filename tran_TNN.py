import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import json

max_token = 30

def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

# lấy dữ liệu đào tạo
x = read_sentences_from_file('x.ta')
y = read_sentences_from_file('y.ta')

# đọc file word_list
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.word_index = word_index

# mã hóa các câu
X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X, maxlen=max_token)

# chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# chuyển đổi y thành array
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

# hàm tạo transformer
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

# xây dựng mô hình transformer
input_layer = Input(shape=(max_token,))
embedding_layer = Embedding(input_dim=10000, output_dim=128, input_length=max_token)(input_layer)
x = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# điều kiện dừng huấn luyện
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# huấn luyện mô hình
model.fit(X_train, y_train, epochs=60, batch_size=2, validation_data=(X_test, y_test), callbacks=[early_stopping])

# đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.4f}')

# lưu trọng số và bias
model.save_weights('transformer_model.weights.h5')
