import os
import numpy as np
import joblib
from utils import extract_block_features, create_model, write_to_file, delete_if_exists
from extract_features import extract_geometric_features

def apply_model(pdf_path):
    print("Loading model and scaler...")
    model = create_model()
    model.load_weights('save.weights.h5')
    scaler = joblib.load('scaler.save')

    print(f"Processing PDF: {pdf_path}")
    features_data = extract_geometric_features(pdf_path)
    pages = {}
    for data in features_data:
        pages.setdefault(data['page'], []).append(data)

    block_types = ['Header', 'Body', 'Footer', 'Quote']
    delete_if_exists('output.txt')

    for page_number, page_blocks in pages.items():
        print(f"Page {page_number + 1} with {len(page_blocks)} blocks")
        for i, block in enumerate(page_blocks):
            if block.get('num_lines', 0) <= 0:
                continue
            try:
                block_features = np.array(extract_block_features(block)).reshape(1, -1)
                normalized_features = scaler.transform(block_features)
                predicted_class = np.argmax(model.predict(normalized_features)[0])
                predicted_block_type = block_types[predicted_class]
                print(f"Predicted: {predicted_block_type} - Block {i + 1}, Page {page_number + 1}")
                write_to_file(block['raw_block'][4], predicted_block_type)
            except Exception as e:
                print(f"Error processing block {i+1} on page {page_number + 1}: {e}")
                continue

if __name__ == '__main__':
    pdf_path = 'input.pdf'  # Replace with your PDF file path
    apply_model(pdf_path)

