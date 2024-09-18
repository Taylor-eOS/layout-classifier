import string
from math import log2
import fitz
import spacy
import numpy as np
from wordfreq import word_frequency
from sklearn.preprocessing import KBinsDiscretizer
nlp = spacy.load("en_core_web_sm")
from utils import delete_if_exists

def extract_geometric_features(pdf_path):
    delete_if_exists("readout.txt")
    doc = fitz.open(pdf_path)
    page_data = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        all_relative_font_sizes = calculate_all_relative_font_sizes(page)

        for idx, block in enumerate(blocks):
            if len(block) < 6:
                print_to_file(f"Warning: Block at index {idx} on page {page_num + 1} is incomplete")
                #continue
            x0, y0, x1, y1, text, block_id = block[:6]
            if text.strip():
                height = calculate_height(y0, y1)
                width = calculate_width(x0, x1)
                position = calculate_position(y0, page.rect.height)
                letter_count = calculate_letter_count(text)
                punctuation_count = calculate_non_latin_proportion(text)
                total_characters = len(text)
                punctuation_proportion = punctuation_count / total_characters if total_characters > 0 else 0
                average_font_size = calculate_average_font_size(page, idx, letter_count)
                relative_font_size = all_relative_font_sizes[idx]
                num_lines = calculate_num_lines(page, idx)
                average_word_length = calculate_average_word_length(text)
                average_words_per_sentence = calculate_average_words_per_sentence(text)
                starts_with_number = calculate_starts_with_number(text)
                capitalization_proportion = calculate_capitalization_proportion(text)
                average_word_commonality = get_word_commonality(text)
                entropy_value = calculate_entropy(text)
                squared_entropy = entropy_value ** 2
                lexical_density = calculate_lexical_density(text)

                page_data.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "height": height,
                    "width": width,
                    "position": position,
                    "letter_count": letter_count,
                    "font_size": average_font_size,
                    "relative_font_size": relative_font_size,
                    "num_lines": num_lines,
                    "punctuation_proportion": punctuation_proportion,
                    "average_word_length": average_word_length,
                    "average_words_per_sentence": average_words_per_sentence,
                    "starts_with_number": starts_with_number,
                    "capitalization_proportion": capitalization_proportion,
                    "average_word_commonality": average_word_commonality,
                    "squared_entropy": squared_entropy,
                    "lexical_density": lexical_density,
                    "block_number_on_page": idx + 1,
                    "page": page_num,
                    "raw_block": block
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

def calculate_non_latin_proportion(text):
    if not isinstance(text, str):
        try:
            text = ''.join(text)
        except TypeError:
            warnings.warn("Input is neither a string nor an iterable of strings. Returning 0.", UserWarning)
            return 0
    latin_letters = set(string.ascii_letters)
    non_latin_count = sum(1 for c in text if c not in latin_letters)
    total_characters = len(text)
    if total_characters == 0:
        return 0
    proportion = non_latin_count / total_characters
    return cap_at_one(proportion)

def calculate_average_font_size(page, block_index, letter_count):
    blocks_dict = page.get_text("dict").get("blocks", [])
    if block_index < 0 or block_index >= len(blocks_dict):
        return 0
    block = blocks_dict[block_index]
    lines = block.get("lines", [])
    if not lines:
        return None
    font_sizes = []
    for line in lines:
        spans = line.get("spans", [])
        for span in spans:
            if "size" in span:
                font_sizes.append(span["size"])
    return sum(font_sizes) / len(font_sizes) if font_sizes else None

def calculate_all_relative_font_sizes(page):
    blocks_dict = page.get_text("dict").get("blocks", [])
    all_font_sizes = []
    for idx, block in enumerate(blocks_dict):
        letter_count = len(block['text']) if 'text' in block else 0
        avg_font_size = calculate_average_font_size(page, idx, letter_count)
        if avg_font_size is not None:
            all_font_sizes.append(avg_font_size)
    max_font_size = max(all_font_sizes) if all_font_sizes else 1
    all_relative_font_sizes = [font_size / max_font_size for font_size in all_font_sizes]
    return all_relative_font_sizes

def calculate_num_lines(page, block_index):
    blocks_dict = page.get_text("dict").get("blocks", [])
    if block_index < 0 or block_index >= len(blocks_dict):
        return 0
    block = blocks_dict[block_index]
    lines = block.get("lines", [])
    if not lines:
        return 0
    return len(lines)

def calculate_average_word_length(text):
    words = [word.strip(string.punctuation) for word in text.split()]
    word_lengths = [len(word) if word.isalpha() else 1 for word in words]  
    return sum(word_lengths) / len(word_lengths) if word_lengths else 0

def calculate_average_words_per_sentence(text):
    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    return sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

def calculate_starts_with_number(text):
    stripped_text = text.strip()
    if not stripped_text:
        return 0
    first_char = stripped_text[0]
    return 1 if first_char.isdigit() else 0

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

def calculate_entropy(text):
    if not text:
        return 0
    probabilities = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * log2(p) for p in probabilities if p > 0)

def calculate_lexical_density(text):
    doc = nlp(text)
    total_words = len([token for token in doc if token.is_alpha])
    content_words = [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
    if total_words == 0:
        return 0
    densi = len(content_words) / total_words
    return cap_at_one(densi)

def cap_at_one(value):
    if isinstance(value, (int, float)):
        return min(1, value)
    else:
        warnings.warn("Unrecognized number format. Returning 0.", UserWarning)
        return 0

def print_to_file(value):
    with open("readout.txt", "a", encoding='utf-8') as file:
            file.write(f"{value}\n")
            print(f"{value}")

