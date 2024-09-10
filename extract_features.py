import fitz  #PyMuPDF
import string
from wordfreq import word_frequency #pip install wordfreq

def extract_geometric_features(pdf_path):
    doc = fitz.open(pdf_path)
    page_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block_no, block in enumerate(blocks):
            x0, y0, x1, y1, text, block_no = block[:6]
            
            if text.strip(): #Ignore blocks without text (or only whitespace)
                height = calculate_height(y0, y1)
                width = calculate_width(x0, x1)
                position = calculate_position(y0, page.rect.height)
                letter_count = calculate_letter_count(text)
                punctuation_count = calculate_punctuation_count(text)
                total_characters = len(text)
                punctuation_proportion = punctuation_count / total_characters if total_characters > 0 else 0
                average_font_size = calculate_average_font_size(page, block_no)
                num_lines = calculate_num_lines(page, block_no)
                average_word_length = calculate_average_word_length(text)
                average_words_per_sentence = calculate_average_words_per_sentence(text)
                starts_with_number = calculate_starts_with_number(text)
                capitalization_proportion = calculate_capitalization_proportion(text)
                average_word_commonality = get_word_commonality(text)

                page_data.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "height": height,
                    "width": width,
                    "position": position,
                    "letter_count": letter_count,
                    "font_size": average_font_size,
                    "num_lines": num_lines,
                    "punctuation_proportion": punctuation_proportion,
                    "average_word_length": average_word_length,
                    "average_words_per_sentence": average_words_per_sentence,
                    "starts_with_number": starts_with_number,
                    "capitalization_proportion": capitalization_proportion,
                    "average_word_commonality": average_word_commonality,
                    "block_number_on_page": block_no + 1,
                    "page": page_num
                })
    return page_data

def calculate_height(y0, y1):
    return y1 - y0

def calculate_width(x0, x1):
    return x1 - x0

def calculate_position(y0, page_height):
    return y0 / page_height

def calculate_letter_count(text):
    return sum(c.isalpha() for c in text)

def calculate_punctuation_count(text):
    return sum(1 for c in text if c in '.,;:!?/—1234567890"()-')

def calculate_average_font_size(page, block_no):
    spans = page.get_text("dict")["blocks"][block_no]["lines"][0]["spans"]
    font_sizes = [span["size"] for span in spans]
    return sum(font_sizes) / len(font_sizes) if font_sizes else 0

def calculate_num_lines(page, block_no):
    return len(page.get_text("dict")["blocks"][block_no]["lines"])

def calculate_average_word_length(text):
    words = [word.strip(string.punctuation) for word in text.split()]
    word_lengths = [len(word) if word.isalpha() else 1 for word in words]  #Counts non-alphabetic tokens as 1
    return sum(word_lengths) / len(word_lengths) if word_lengths else 0

def calculate_average_words_per_sentence(text):
    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

def calculate_starts_with_number(text):
    first_char = text.strip()[0]
    if first_char.isdigit(): #first_char.isalpha()
        return 1
    else:
        return 0

def calculate_capitalization_proportion(text):
    letter_count = sum(1 for c in text if c.isalpha())
    capitalized_count = sum(1 for c in text if c.isupper())
    return capitalized_count / letter_count if letter_count > 0 else 0

def get_word_commonality(text, scale_factor=100):
    words = [word.strip(string.punctuation).lower() for word in text.split() if word.isalpha()]
    if not words:
        return 0.01
    word_frequencies = [word_frequency(word, 'en') for word in words if word_frequency(word, 'en') > 0]
    if not word_frequencies:
        return 0.01
    avg_frequency = sum(word_frequencies) / len(word_frequencies)
    return avg_frequency * scale_factor
