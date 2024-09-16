import os
import numpy as np
import tensorflow as tf
import csv
from extract_features import extract_geometric_features
from gui import show_pdf_page_with_block
from sklearn.preprocessing import StandardScaler
from utils import create_model, extract_block_features, save_weights, write_features

scaler = StandardScaler()

def main():
    print(f"Started")
    pdf_path = "input.pdf"
    features_data = extract_geometric_features(pdf_path)
    model = create_model()
    input_weights_file = 'load.weights.h5'
    if os.path.exists(input_weights_file):
        model.load_weights(input_weights_file)
    pages = {}

    csv_file = "block_features.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)

    for data in features_data:
        if data['page'] not in pages:
            pages[data['page']] = []
        pages[data['page']].append(data)

    all_blocks_features = []
    for page_blocks in pages.values():
        for block in page_blocks:
            all_blocks_features.append(extract_block_features(block))
    
    all_blocks_features = np.array(all_blocks_features)
    scaler.fit(all_blocks_features)
    block_batch = []
    label_batch = []

    for page_number, page_blocks in pages.items():
        print(f"Page {page_number + 1}")

        for i, block in enumerate(page_blocks):
            block_features = np.array(extract_block_features(block)).reshape(1, -1)
            normalized_features = scaler.transform(block_features)
            block_batch.append(normalized_features)
            predicted_class = np.argmax(model.predict(normalized_features)[0])
            print(f"{['Header', 'Body', 'Footer', 'Quote'][predicted_class]}")
            correct_label = show_pdf_page_with_block(pdf_path, block, predicted_class, page_number)
            
            if correct_label is None or correct_label == 4:
                print("No valid class selected or GUI closed. Skipping model update.")
                continue

            write_features(csv_file, extract_block_features(block))

            y_train = [1 if j == correct_label else 0 for j in range(4)]
            label_batch.append(y_train)

            update_csv_with_block_type(csv_file, correct_label)

            if len(block_batch) == 5:
                block_batch = np.vstack(block_batch)
                label_batch = np.array(label_batch)
                model.fit(block_batch, label_batch, epochs=1)
                output_weights_file = 'save.weights.h5'
                save_weights(model, output_weights_file)

                block_batch = []
                label_batch = []

    if len(block_batch) > 0:
        block_batch = np.vstack(block_batch)
        label_batch = np.array(label_batch)
        model.fit(block_batch, label_batch, epochs=1)
        save_weights(model, output_weights_file)

def update_csv_with_block_type(csv_file, correct_label):
    temp_file = "temp.csv"
    block_type = ['Header', 'Body', 'Footer', 'Quote'][correct_label]

    with open(csv_file, mode='r') as infile, open(temp_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        headers = next(reader)
        if "block_type" not in headers:
            headers.append("block_type")
        writer.writerow(headers)

        for row in reader:
            if len(row) == len(headers) - 1:
                row.append(block_type)
            writer.writerow(row)

    os.replace(temp_file, csv_file)

if __name__ == "__main__":
    main()

