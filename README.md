# metodecnn
```python

# Install library jika belum (biasanya sudah di Colab)
# !pip install tensorflow scikit-learn pandas

# Import library
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Upload file CSV ke Google Colab
from google.colab import files
uploaded = files.upload()

# Cek nama file yang diupload, pastikan data_berita.csv
import io
data = pd.read_csv(io.BytesIO(uploaded['data_berita.csv']))

# Lihat isi dataset
print(data.head())

# Ambil teks dan label
texts = data['text'].values
labels = data['label'].values

# Tokenisasi teks
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequence (agar panjang sama)
maxlen = 100
X = pad_sequences(sequences, maxlen=maxlen)
y = labels

# Split data ke training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bangun model CNN
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=4)

# Evaluasi model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nHasil Evaluasi:")
print(classification_report(y_test, y_pred))
