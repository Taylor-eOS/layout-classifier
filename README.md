## Machine Learning Layout Classifier

### Overview

This project is a simple educational experiment for parcticing machine learning concepts. It **extracts and classifies text blocks** from PDF documents based on their geometric and linguistic features. The program processes individual blocks of text, such as headers, body text, and footnotes, and uses machine learning techniques to predict the type of text. The project is designed to assist in the automated analysis of PDF documents by leveraging both textual and structural features.

The project is currently a work in progress, and does not posess the ability to apply the model to whole documents yet.

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
- **GUI**: Provides a simple interface for user input and model validation during document analysis.
