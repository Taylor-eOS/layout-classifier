import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from extract_features import extract_geometric_features
from gui import show_pdf_page_with_block
from utils import create_model, extract_block_features, save_weights, write_features, drop_to_file, delete_if_exists

def main(test_mode=False, test_file='test.csv'):
    print("Starting...")
    pdf_path = "input.pdf"
    features_data = extract_geometric_features(pdf_path)
    model = create_model()
    input_weights_file = 'load.weights.h5'
    if os.path.exists(input_weights_file):
        model.load_weights(input_weights_file)
        print(f"Loading weights from file!")

    pages = {}
    csv_file = "block_features.csv"
    delete_if_exists(csv_file)
    delete_if_exists("output.txt")
    for data in features_data:
        pages.setdefault(data['page'], []).append(data)
    all_blocks_features = [extract_block_features(block) for blocks in pages.values() for block in blocks]
    scaler = StandardScaler().fit(all_blocks_features)
    joblib.dump(scaler, 'scaler.save')
    block_batch = []
    label_batch = []

    if test_mode:
        try:
            test_file_reader = open(test_file, 'r')
            print(f"Opened {test_file} for reading")
        except Exception as e:
            print(f"Failed to open {test_file}: {e}")
            return

    correct_predictions = 0
    total_predictions = 0
    total_blocks = 0
    block_types = ['Header', 'Body', 'Footer', 'Quote']

    for page_number, page_blocks in pages.items():
        print(f"Page {page_number + 1} with {len(page_blocks)} blocks")
        total_blocks += len(page_blocks)

        for i, block in enumerate(page_blocks):
            if block.get('num_lines', 0) <= 0:
                continue
            try:
                block_features = np.array(extract_block_features(block)).reshape(1, -1)
                normalized_features = scaler.transform(block_features)
                predictions = model.predict(normalized_features)[0]
                predicted_class = np.argmax(predictions)
                predicted_block_type = block_types[predicted_class]
                certainty = predictions[predicted_class]
        
                print(f"{predicted_block_type} - Block {i + 1}, Page {page_number + 1}, Certainty: {certainty:.2f}")

                if test_mode:
                    correct_label_line = test_file_reader.readline().strip()
                    if not correct_label_line:
                        print(f"Error: Ran out of labels at block {total_predictions + 1}")
                        test_file_reader.close()
                        return
                    try:
                        correct_label = int(correct_label_line)
                    except ValueError:
                        print(f"Error: Non-integer label '{correct_label_line}' at block {total_predictions + 1}")
                        test_file_reader.close()
                        return

                    if correct_label < 0 or correct_label > 3:
                        print(f"Error: Invalid label {correct_label} at block {total_predictions + 1}")
                        test_file_reader.close()
                        return

                    total_predictions += 1

                    if correct_label == predicted_class:
                        correct_predictions += 1
                    block_label = correct_label

                else:
                    correct_label = show_pdf_page_with_block(pdf_path, block, predicted_class, page_number)
                    if correct_label is None or correct_label not in [0, 1, 2, 3]:
                        print("Error: Label could not be determined or is out of range")
                        return
                    block_label = correct_label

                drop_to_file(block['raw_block'][4], block_types[block_label])
                is_correct = (predicted_class == correct_label)

                block_batch.append(normalized_features[0])
                y_train = [1 if j == block_label else 0 for j in range(4)]
                label_batch.append(y_train)

                write_features(
                    csv_file,
                    extract_block_features(block),
                    block_type=block_types[block_label],
                    is_correct=is_correct,
                    predicted_block_type=predicted_block_type,
                    certainty=certainty
                )

                if len(block_batch) == 5:
                    block_batch_np = np.array(block_batch)
                    label_batch_np = np.array(label_batch)
                    if len(block_batch_np) == len(label_batch_np):
                        model.fit(block_batch_np, label_batch_np, epochs=1)
                        if not test_mode:
                            save_weights(model, 'save.weights.h5')
                    block_batch = []
                    label_batch = []

            except Exception as e:
                print(f"Error while processing block {i+1} on page {page_number + 1}: {e}")
                if test_mode:
                    test_file_reader.close()
                return

    if block_batch and len(block_batch) == len(label_batch):
        block_batch_np = np.array(block_batch)
        label_batch_np = np.array(label_batch)
        model.fit(block_batch_np, label_batch_np, epochs=1)

    if test_mode:
        save_weights(model, 'save.weights.h5')
        print(f"Total blocks processed: {total_blocks}")
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"Test Accuracy: {accuracy:.2f}%")
        else:
            print("No predictions made during the test.")
        test_file_reader.close()

if __name__ == "__main__":
    import sys
    test_mode = '--test' in sys.argv
    print("Running in test mode" if test_mode else "Running in normal mode")
    try:
        main(test_mode=test_mode)
    except Exception as e:
        print(f"An error occurred: {e}")

