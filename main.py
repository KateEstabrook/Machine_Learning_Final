import ast
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import shutil
import string
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers
from tensorflow.keras import losses


# very cool website, much wow
# https://www.tensorflow.org/tutorials/keras/text_classification

#print(tf.__version__)

df = pd.read_csv('games.csv')

def str_to_lst(value):
    # If there are no genres
    if pd.isna(value):
        return []
    # Change genres to a list and return
    parsed_list = ast.literal_eval(value)
    return parsed_list
    

# For each column, take the string of genres and change it to a list 
df['genres'] = df['genres'].apply(str_to_lst)
print(df['genres'].head())

# for val in df.at[0, 'genres']:
#     print(val)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

genre_classes = mlb.classes_

X = df['detailed_description'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

max_features = 20000
sequence_length = 300

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(X_train.values)


batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test))

train_ds = (
    train_ds
    .map(lambda x, y: (vectorize_layer(x), y))
    .cache()
    .shuffle(10000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .map(lambda x, y: (vectorize_layer(x), y))
    .batch(batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

embedding_dim = 128
num_genres = len(genre_classes)

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(num_genres, activation='sigmoid')  # sigmoid → multi-label
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)
