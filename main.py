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

sample_text = [
    "Traditional Japanese mahjong game with lovely ACG characters, and all kinds of anime memes. You can play Riichi mahjong game anytime anywhere, enjoy the real mahjong experience! Come and join Riichi City! [Riichi City Features] ‚óÜFree to Play and Fair Environment Enjoy different modes and features without any cost! Indulge in the real Riichi mahjong games with your friends anywhere, anytime! Experience truly random tiles, without any additional adjustments! Chii! Pon! Kan! Pinfu! Chinitsu! Kokushi Musou! Explore a wide range of yaku combinations that you love in a fair and secure gaming environment! ‚óÜReal-Life Design Elements The real-life design elements in the modern Riichi City, such as mahjong houses, caf√©s, convenience stores and more, bring the realness of playing live Japanese mahjong games to the app! Experience the fierce competition in the fast-paced Japanese mahjong matches! ‚óÜACG Characters with Unique Playing Styles &amp; Personalities Meet and bond with the ACG characters in-app, each featuring their own playing styles and personalities, ranging from aggressive to defensive and skilled at reading discards... You name it!You can choose different waifu to grow with them and take your Japanese mahjong skills to the next level! ‚óÜFully Voiced by Famous Japanese CVs Renowned Japanese voice actors and actresses bring the ACG characters in Riichi City to life with their exceptional voice acting talent! Lead them as their captain, start competing against others in the mahjong tournaments, and enjoy your game with these mahjong stars! CVs: Ogura Yui, Namikawa Daisuke, Ishikawa Yui, Arai Satomi, Suguta Hina, Morishima Shuuta, Hasegawa Reina... ‚óÜBeginner-Friendly Functions Riichi City has a lot to offer, including tutorials, game log, spectator mode, hints for the winning tiles, tips for drawing and discarding tiles, auto hints, point calculation, and much more! All of these features are designed to help beginners ease into the Riichi mahjong world! ‚óÜCommunity and Abundant Events with Generous Rewards No longer play alone! Join the vibrant community and communicate with mahjong players from around the world! Stay tuned for the official knockout tournaments, Japanese mahjong matches, limited-time challenges, and events in the future, all featuring tons of prizes! Customer service: riichicitysupport@mahjong-jp.com ",
    "Special Offer About the Game The conquistador, who has faced many trials in his life, decided to retire and become a hermit. Just lying in his solitary cave away from people. But it seems that fate itself throws new challenges at you even in such a simple matter. The coffin in which you like to sleep has disappeared. You need to embark on a search and solve a multitude of puzzles until you find it. This is how its story begins. There are 4 levels in which you will understand the backstory of the Conquistador and get the answer to why he went on the quest. Plot: You are about to embark on a search for the missing item and find it. Puzzles: Inventive and challenging puzzles will help you spend many hours solving captivating logical riddles. Graphics: A distinctive hand-drawn artistic style will allow you to enjoy a unique and charming aesthetic. Sound: Ambient sounds, background music, and character-specific sound effects will further immerse you in the game. Dialogues: Conquistador relies on non-verbal storytelling. So, no long and boring dialogues - only expressive animation of characters and the surrounding environment."
]

preds = model.predict(vectorize_layer(sample_text))

# Convert probabilities → binary predictions
pred_labels = (preds > 0.5).astype(int)
decoded = mlb.inverse_transform(pred_labels)
print(decoded)