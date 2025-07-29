import os 
from pymupdf4llm import to_markdown

PDF_DIR = "./pdf_files/"

def extract_pdf():
    all_text = ""
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Loading: {pdf_path}")
            md_text = to_markdown(pdf_path)
            all_text += md_text + "\n"
    return all_text

if __name__ == "__main__":
    text = extract_pdf()
    print(text)
    with open("./text_document/extracted_text.txt", "w") as f:
        f.write(text)
    print("Text extraction complete. Check 'extracted_text.txt' for the output.")
