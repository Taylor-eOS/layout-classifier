import os
import random
from PyPDF2 import PdfReader, PdfWriter

def assemble_random_pages(input_folder, output_file, pages_per_pdf=5, skip_first_n=5):
    pdf_filenames = [file_name for file_name in os.listdir(input_folder) if file_name.endswith(".pdf")]

    if not pdf_filenames:
        print("No PDF files found in the folder.")
        return

    selected_pages = []

    for _ in range(pages_per_pdf):
        random.shuffle(pdf_filenames)
        for file_name in pdf_filenames:
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}")
            try:
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    total_pages = len(reader.pages)
                    if total_pages > skip_first_n:
                        available_pages = list(range(skip_first_n, total_pages))
                    else:
                        available_pages = list(range(total_pages))
                    if available_pages:
                        page_num = random.choice(available_pages)
                        selected_pages.append((file_path, page_num))
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue  

    random.shuffle(selected_pages)
    pdf_writer = PdfWriter()
    for file_path, page_num in selected_pages:
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                page = reader.pages[page_num]
                pdf_writer.add_page(page)
                print(f"Adding a page from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error adding page from {file_path}: {e}")
            continue
    with open(output_file, "wb") as out_file:
        pdf_writer.write(out_file)

input_folder = "/home/l/deve/pdf_shuffler/pdfs/"
output_file = "assembled_pages.pdf"
pages_per_pdf = 2  
skip_first_n = 20   

assemble_random_pages(input_folder, output_file, pages_per_pdf, skip_first_n)

