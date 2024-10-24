import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

max_token = 30

# tải word-list
with open('word_index.json', 'r') as json_file:
    word_index = json.load(json_file)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.word_index = word_index

# xây dựng lại mô hình transformer (phải khớp với mô hình đã huấn luyện)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    # feed-forward layer
    ff_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return ff_output

# tạo mô hình Transformer
input_layer = tf.keras.layers.Input(shape=(max_token,))
embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=max_token)(input_layer)
x = transformer_encoder(embedding_layer, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# tải trọng số đã lưu vào mô hình transformer
model.load_weights('transformer_model.weights.h5')

# câu cần dự đoán
input_sentence = "i love my cat, my cat is so dump and i like this"

# Mã hóa câu
sequence = tokenizer.texts_to_sequences([input_sentence])
padded_sequence = pad_sequences(sequence, maxlen=max_token)

# dự đoán
predictions = model.predict(padded_sequence)
print(f'Predicted Probabilities: {predictions}')

# in kết quả dự đoán
predicted_class = np.argmax(predictions, axis=1)  # Lấy chỉ số của lớp có xác suất cao nhất
print(f'Predicted Class: {predicted_class}')
