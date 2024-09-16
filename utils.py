import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import threading

def create_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(15,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_block_features(block):
    features = [
        block["height"], block["width"], block["position"], block["letter_count"], block["font_size"],
        block["num_lines"], block["punctuation_proportion"], block["average_word_length"],
        block["average_words_per_sentence"], block["starts_with_number"], block["capitalization_proportion"],
        block["average_word_commonality"], block["block_number_on_page"], block["squared_entropy_value"],
        block["lexical_density_value"]
    ]
    return features

def save_weights(model, file_path):
    def save_task():
        model.save_weights(file_path, overwrite=True)
    threading.Thread(target=save_task).start()

def write_features(file_path, block_features):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "height", "width", "position", "letter_count", "font_size", 
                "num_lines", "punctuation_proportion", "average_word_length", 
                "average_words_per_sentence", "starts_with_number", 
                "capitalization_proportion", "average_word_commonality", 
                "block_number_on_page", "squared_entropy_value", "lexical_density_value"
            ])
        writer.writerow(block_features)

