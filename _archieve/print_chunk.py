from chunk_text import split_text_with_langchain

# Load text from file
with open("./text_document/ocr_all_pdf.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Split into chunks
chunks = split_text_with_langchain(full_text, chunk_size=1024, chunk_overlap=100)

# Print each chunk with index
for idx, chunk in enumerate(chunks):
    print(f"--- Chunk {idx} ---\n{chunk}\n")
