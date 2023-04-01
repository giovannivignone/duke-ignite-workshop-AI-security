import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

# Load and preprocess data
def preprocess_data(file_path):
    # Read the CSV file
    dataset = pd.read_csv(file_path)

    # Drop unnecessary columns
    dataset = dataset.drop(['frame.number', 'frame.time'], axis=1)

    # One-hot encoding for categorical features
    categorical_features = ['eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.proto']
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(dataset[categorical_features])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(categorical_features))

    # Merge one-hot encoded features back into the dataset
    dataset = pd.concat([dataset.drop(categorical_features, axis=1), onehot_encoded_df], axis=1)

    # Normalize numerical features
    numerical_features = ['frame.len', 'ip.len', 'tcp.len', 'tcp.srcport', 'tcp.dstport', 'Value']
    scaler = StandardScaler()
    dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

    return dataset

# Prepare data for training
def prepare_data(dataset):
    # Separate features and labels
    X = dataset.drop('label', axis=1)
    y = dataset['label']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder

# Build and train neural network model
def build_and_train_model(X_train, y_train):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 output classes: normal, suspicious, malicious
    ])

    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')

# Main function
def main():
    file_path = 'Preprocessed_data.csv'
    dataset = preprocess_data(file_path)
    X_train, X_test, y_train, y_test, encoder = prepare_data(dataset)
    model = build_and_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
    pass