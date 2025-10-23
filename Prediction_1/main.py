import ast
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers

def str_to_lst(value):
    """Convert stringified list of genres into a Python list."""
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except Exception:
        return []

if __name__ == "__main__":
    # Load and preprocess the dataset
    df = pd.read_csv('games.csv')

    df['genres'] = df['genres'].apply(str_to_lst)
    print("Sample parsed genres:\n", df['genres'].head())

    # Prepare features and labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genres'])
    genre_classes = mlb.classes_

    # Ensure no empty values in description
    X = df['detailed_description'].astype(str)
    game_names = df['name'].astype(str) if 'name' in df.columns else [f"Game_{i}" for i in range(len(df))]

    # 90/10 train-test split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, game_names, test_size=0.1, shuffle=True
    )

    # Text vectorization
    max_features = 20000
    sequence_length = 300

    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    vectorize_layer.adapt(X_train.values)

    # Prepare TensorFlow datasets
    batch_size = 32

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train.values, y_train))
        .map(lambda x, y: (vectorize_layer(x), y))
        .cache()
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test.values, y_test))
        .map(lambda x, y: (vectorize_layer(x), y))
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Build and train model
    embedding_dim = 128
    num_genres = len(genre_classes)

    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        # Use sigmoid for multi-label
        layers.Dense(num_genres, activation='sigmoid') 
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

    # Evaluate and make Predictions
    preds = model.predict(vectorize_layer(X_test.values))
    pred_labels = (preds > 0.5).astype(int)
    decoded = mlb.inverse_transform(pred_labels)

    # Save results to text file
    output_file = "Prediction_1/Predictions.txt"

    # Convert back to actual genre labels
    true_genres_decoded = mlb.inverse_transform(y_test)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(X_test)):
            name = names_test.iloc[i]
            # Get real genres
            if len(true_genres_decoded[i]) == 1:
                genres_real = f"(\"{true_genres_decoded[i][0]}\")"
            elif true_genres_decoded[i]:
                genres_real = ""
                for gen in true_genres_decoded[i]:
                    if gen == true_genres_decoded[i][-1]:
                        genres_real += f"\"{gen}\""
                    else:
                        genres_real += f"\"{gen}\", "
                genres_real = "(" + genres_real + ")"
            else:
                genres_real = "\"no genres\""

            # Get outputted genres
            if len(decoded[i]) == 1:
                genres = f"(\"{decoded[i][0]}\")"
            elif decoded[i]:
                genres = ""
                for gen in decoded[i]:
                    if gen == decoded[i][-1]:
                        genres += f"\"{gen}\""
                    else:
                        genres += f"\"{gen}\", "
                genres = "(" + genres + ")"
            else:
                genres = "\"no genres predicted\""
            
            # Write name and genres to file
            f.write(f"\"{name}\", {genres}, {genres_real}\n")

    # Export trained model
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model
    ])

    # Build before saving
    export_model(tf.constant(["sample text"]))
    export_model.save("Prediction_1/model.keras")

    # Save label binarizer
    joblib.dump(mlb, "Prediction_1/binarizer.pkl")
