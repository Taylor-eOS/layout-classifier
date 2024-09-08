import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from extract_features import extract_geometric_features
from gui import show_pdf_page_with_block

def create_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_global_features(blocks):
    total_features = np.array([
        [block["height"], block["letter_count"], block["font_size"], block["num_lines"], block["punctuation_proportion"]]
        for block in blocks
    ])
    return np.mean(total_features, axis=0)

def format_features(features):
    return " ".join(f"{feature:.1f}" for feature in features)

def main():
    pdf_path = "your_pdf_file.pdf"
    features_data = extract_geometric_features(pdf_path)
    model = create_model()
    pages = {}
    for data in features_data:
        if data['page'] not in pages:
            pages[data['page']] = []
        pages[data['page']].append(data)
    for page_number, page_blocks in pages.items():
        print(f"Processing Page {page_number + 1}")
        global_means = calculate_global_features(page_blocks)
        print(f"Global: {format_features(global_means)}")
        for i, block in enumerate(page_blocks):
            normalized_features = np.array([
                block["height"] / global_means[0],
                block["letter_count"] / global_means[1],
                block["font_size"] / global_means[2],
                block["num_lines"] / global_means[3],
                block["punctuation_proportion"] / global_means[4]
            ])
            print(f"Local: {format_features(normalized_features)}")
            block_features = normalized_features.reshape(1, 5)
            pred = model.predict(block_features)
            predicted_class = np.argmax(pred[0])
            print(f"Predicted Class: {['Header', 'Body', 'Footer'][predicted_class]}")
            correct_label = show_pdf_page_with_block(pdf_path, block, predicted_class, page_number)
            if correct_label is None or correct_label == 3:
                print("No valid class selected or GUI closed. Skipping model update.")
            else:
                y_train = [1 if j == correct_label else 0 for j in range(3)]
                model.fit(block_features, np.array([y_train]), epochs=1)
                global_means = calculate_global_features(page_blocks)
                print(f"Global: {format_features(global_means)}")

if __name__ == "__main__":
    main()

