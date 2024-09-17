def write_to_file(block_text):
    with open("output.txt", "a") as file:
        file.write(f"{block_text}\n\n")
