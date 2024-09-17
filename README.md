## Machine Learning Layout Classifier

### Overview

This project is a simple educational experiment for practicing machine learning concepts. It **extracts and classifies text blocks** from PDF documents based on their geometric and linguistic features. The program processes individual blocks of text, such as headers, body text, and footnotes, and uses machine learning techniques to predict the type of text. The project is designed to assist in the automated classification of PDF documents, for instance for conversion into reflowable formats.

The project is currently a work in progress. It does posess a simple ability to apply previously trained models to documents, but this requires a fair amount of training data to give accurate results. Machine learning is just not good at specificity. I'm seeing 99% accurate results for everything except quotes after about 100 pages of training on a specific file, which would still save you a quarter of the work if your file is 400 pages long.

### Key Features

1. **Geometric Feature Extraction**: 
   The program identifies and extracts geometric features from PDF text blocks, including dimensions such as height, width, position on the page, and formatting details like font size and the number of lines.

2. **Linguistic Feature Extraction**:
   The system analyzes the linguistic properties of each block, including average word length, word commonality (the frequency of words in standard English usage), and punctuation density. 

3. **Classification Model**:
   A machine learning model, built using TensorFlow, predicts the category of each text block. The categories include Header, Body, Footer, and Quote. The model is trained iteratively with user input to improve classification accuracy over time.

4. **User Input Integration**:
   A GUI allows users to validate and correct the predicted classifications by reviewing text blocks and providing feedback. This input is used to continuously train the classification model for improved performance.

5. **CSV Output for Analysis**:
   Each run generates a CSV file containing the extracted features of text blocks, which can be used for further analysis or auditing. The CSV file is reset on each run to avoid accumulating redundant data.

5. **Weights Loading**:
   Model weights are saved automatically and can be loaded on launch.

### Use Cases

- **Education**: 
  The main aim is to serve as a vehicle for experienceing machine learning concepts on practical, functional examples.
  
- **Document Structure Analysis**: 
  Automatically determine the structure of a PDF by classifying text blocks, which can be useful in document indexing, content extraction, or conversion processes.
  
- **Text-Type Differentiation**: 
  The system differentiates between various types of text, such as footnotes, body text, and headers, by analyzing geometric and linguistic characteristics.

- **Text Extracton**: 
  Potentially the tool can be used to selectively extract text from documents, for instance as a processing step in combination with other tools like OCR.
  

### Technology

- **Python**: The core language for feature extraction, text processing, and machine learning.
- **TensorFlow**: Used to build and train the text classification model.
- **Scikit-learn**: For standardizing input features using the `StandardScaler`.
- **Pygame**: Provides a simple interface for user input and model validation during document analysis.

---

## How to Use

### 1. **Project Setup**
Before running the scripts, make sure you have the necessary dependencies installed. The project uses `PyMuPDF`, `Python 3.8+`, `TensorFlow`, `scikit-learn`, `joblib`, `wordfreq`, `pygame`, `numpy`, `spacy`, and other utilities for PDF processing.

#### **Install Dependencies**
To install the required dependencies, run `python install_dependencies.py` or check the file import statements.

```bash
pip install tensorflow scikit-learn joblib numpy PyMuPDF wordfreq pygame spacy
```
These lists might be outdated and you can install additional packages as errors demand.

### 2. **Training the Model**
The training process involves extracting geometric features from a sample PDF file, fitting a neural network model, and saving the model weights for future use.

#### **Step 1: Train the Model**
To train the model, use the `train_model.py` script. In the GUI, select which type of block each highlighted piece of text is. The script processes a PDF, extracts text block features, trains the model, and saves both the model weights and scaler for future inference.
```bash
python train_model.py
```

Remane the `save.weights.h5` to `load.weights.h5` for it to get loaded at the launch.

- **Inputs:** A sample PDF (`input.pdf`) that will be processed for training.
- **Outputs:** 
  - `save.weights.h5`: The trained model weights.
  - `scaler.save`: The scaler used to normalize the feature data.
  - `block_features.csv`: A CSV file containing the extracted features and classification labels for each block.
  
### 3. **Applying the Model to New PDFs**
After the model has been trained, you can apply it to classify text blocks in new PDF files.

#### **Step 2: Apply the Model to a New PDF**
To classify text blocks in a new PDF file, use the `apply_model.py` script. This script loads the saved model you created and the scaler, processes a new PDF file, and outputs the text blocks with predicted classifications. To achieve good results, the model had to be trained on the specific layout you use it on or a diverse enough dataset to generalize the features.

- **Inputs:** 
  - A new PDF file (`pdf_path`) that you want to classify.
  - The model weights (`save.weights.h5`) and scaler (`scaler.save`), which should have been saved during training.
- **Outputs:** 
  - `output.txt`: A text file containing the extracted blocks, annotated with tags (`<h1>`, `<body>`, `<footer>`, `<blockquote>`) based on the predicted classification for each block.

### 4. **Test Mode (Optional)**
You can test the model using a predefined set of labels by passing the `--test` flag to the `train_model.py` script. This allows you to evaluate the model’s performance on a specific dataset.
```bash
python train_model.py --test
```

- **Inputs:** 
  - A CSV file (`test.csv`) with the correct labels for each block in numerical format (0-3), one block per line.
- **Outputs:** 
  - Some performance metrics printed to the console.

### 5. **Additional Files**
- **`extract_features.py`:** Contains functions to extract geometric and textual features from each block in the PDF.
- **`gui.py`:** A utility to display PDFs with block highlights, useful for manual classification or review.
- **`utils.py`:** A collection of utility functions, including `create_model()`, `save_weights()`, `extract_block_features()`, and `write_to_file()`.

### 6. **Important Notes**
- Everything is a bit preliminary and manual. Files have to be named the way they are addressed in the script, and customization largely happens internally.
- The output tags in the `output.txt` file can be customized in the `write_to_file()` function if different HTML-like formatting is required.
- To fine-tune the model, adjust the architecture in the `create_model()` function in `utils.py`.
- Ensure that the sample PDF used for training is representative of the type of documents you intend to classify.

---

**Feature wish list:**
Figure out whether some features are counter-productive and might need to be removed: Average Word Length, Average Word Commonality, Block Number on Page, Lexical Density Value, Squared Entropy Value
Add a 5th 'Formula' block type
Handle images as a 6th type and them to a folder for later use
Handle messy PDF blocks that do not have all elements correctly
Possible additional features: Amount of blocks on that page feature
Test out layer depth and amount
Test amount of neurons
Test other neural net variables (optimizer, loss function, metric)
Other activation functions: leaky_relu, elu, tanh, swish
Make GUI respond to keyboard presses
Ask user what teh right answer is if prediction probability is low
Scale per batch?
Make the GUI window not reopen on each click
Size GUI window more appropriately
Skip button for bad blocks
Undo button
Optimize batch size
Optimize learning rate depending on application
