import csv
import os
import numpy as np
import tensorflow as tf
import threading
from tensorflow.keras import layers

def create_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='gelu', input_shape=(16,)),
        layers.Dense(64, activation='gelu'),
        layers.Dense(32, activation='gelu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_block_features(block):
    features = [
        block["height"], block["width"], block["position"], block["letter_count"], block["font_size"],
        block["relative_font_size"], block["num_lines"], block["punctuation_proportion"], block["average_word_length"],
        block["average_words_per_sentence"], block["starts_with_number"], block["capitalization_proportion"],
        block["average_word_commonality"], block["block_number_on_page"], block["squared_entropy"],
        block["lexical_density"]
    ]
    return features

def save_weights(model, file_path):
    def save_task():
        model.save_weights(file_path, overwrite=True)
    threading.Thread(target=save_task).start()

import os
import csv

import os
import csv

def write_features(file_path, block_features, block_type=None, is_correct=None, predicted_block_type=None, certainty=None):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        if not file_exists:
            writer.writerow([
                "height", "width", "position", "letter_count", "font_size", "relative_font_size", 
                "num_lines", "punctuation_proportion", "average_word_length", 
                "average_words_per_sentence", "starts_with_number", 
                "capitalization_proportion", "average_word_commonality", 
                "block_number_on_page", "squared_entropy", "lexical_density",
                "block_type", "predicted_block_type", "correct_prediction", "certainty"
            ])
        def format_value(value):
            if isinstance(value, float):
                return f"{value:.2f}".replace('.', ',')
            return value
        formatted_features = [format_value(f) for f in block_features]
        row = formatted_features + [
            block_type if block_type is not None else '',
            predicted_block_type if predicted_block_type is not None else '',
            is_correct if is_correct is not None else '',
            format_value(certainty) if certainty is not None else ''
        ]
        writer.writerow(row)

def drop_to_file(block_text, block_type):
    with open("output.txt", "a", encoding='utf-8') as file:
        if block_type == 'Header':
            file.write(f"<h1>{block_text}</h1>\n\n")
        elif block_type == 'Body':
            file.write(f"<body>{block_text}</body>\n\n")
        elif block_type == 'Footer':
            file.write(f"<footer>{block_text}</footer>\n\n")
        elif block_type == 'Quote':
            file.write(f"<blockquote>{block_text}</blockquote>\n\n")
        else:
            file.write(f"{block_text}\n\n")

def delete_if_exists(del_file):
    if os.path.exists(del_file):
        os.remove(del_file)

