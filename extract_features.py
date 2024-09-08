import fitz  # PyMuPDF

def extract_geometric_features(pdf_path):
    doc = fitz.open(pdf_path)
    page_data = []

    # Loop through all pages of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")  # Get blocks of text

        for block in blocks:
            x0, y0, x1, y1, text, block_no = block[:6]
            
            # Ignore blocks without text (or only whitespace)
            if text.strip():  # Only process if text is non-empty
                height = y1 - y0
                letter_count = sum(c.isalpha() for c in text)
                punctuation_count = sum(1 for c in text if c in '.,;:!?/—1234567890"()-')
                total_characters = len(text)
                punctuation_proportion = punctuation_count / total_characters if total_characters > 0 else 0
                spans = page.get_text("dict")["blocks"][block_no]["lines"][0]["spans"]
                font_sizes = [span["size"] for span in spans]
                average_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                num_lines = len(page.get_text("dict")["blocks"][block_no]["lines"])

                # Append data including the required coordinates for the GUI
                page_data.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,  # Block coordinates
                    "height": height,
                    "letter_count": letter_count,
                    "font_size": average_font_size,
                    "num_lines": num_lines,
                    "punctuation_proportion": punctuation_proportion,
                    "page": page_num
                })

    return page_data
