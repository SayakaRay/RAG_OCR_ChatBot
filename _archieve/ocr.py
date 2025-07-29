import os
import base64
import json
import asyncio
from enum import Enum
from typing import Optional

from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
client = Mistral(api_key=api_key)


# --- Pydantic Models ---

class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str = Field(..., description="A description of the image.")

class Document(BaseModel):
    language: str = Field(..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'th').")
    summary: str = Field(..., description="A summary of the document.")
    authors: list[str] = Field(..., description="A list of authors who contributed to the document.")


# --- Utilities ---

def encode_pdf(pdf_path: str) -> Optional[str]:
    """Encode the PDF file to a base64 string."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode PDF {pdf_path}: {e}")
        return None


async def process_chunk_async(start_page: int, end_page: int, reader: PdfReader, input_pdf_path: str) -> Optional[dict]:
    """Process a chunk of the PDF asynchronously using Mistral OCR."""
    writer = PdfWriter()
    for i in range(start_page, end_page):
        writer.add_page(reader.pages[i])

    temp_name = f"chunk_{start_page+1}_to_{end_page}.pdf"
    temp_path = os.path.join(os.path.dirname(input_pdf_path), temp_name)

    with open(temp_path, "wb") as f:
        writer.write(f)

    base64_pdf = encode_pdf(temp_path)
    os.remove(temp_path)
    if not base64_pdf:
        return None

    def call_ocr():
        response = client.ocr.process(
            model="mistral-ocr-latest",
            pages=list(range(end_page - start_page)),
            document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
            bbox_annotation_format=response_format_from_pydantic_model(Image),
            document_annotation_format=response_format_from_pydantic_model(Document),
            include_image_base64=True
        )
        data = json.loads(response.model_dump_json())
        data["meta"] = {"start_page": start_page + 1, "end_page": end_page}
        return data

    return await asyncio.to_thread(call_ocr)


async def process_pdf_async(input_pdf_path: str, pages_per_chunk: int = 2) -> list[dict]:
    """Split the PDF into chunks and run OCR on each chunk in parallel."""
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    tasks = []

    for i in range(0, total_pages, pages_per_chunk):
        end = min(i + pages_per_chunk, total_pages)
        tasks.append(process_chunk_async(i, end, reader, input_pdf_path))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def get_all_pdf(path_file: str, output_path: str = "./text_document/ocr_all_pdf.txt"):
    """Main function to extract text + image captions and write to .txt."""
    all_responses = asyncio.run(process_pdf_async(path_file))
    all_pdf = ""

    for response_dict in all_responses:
        for page in response_dict["pages"]:
            # Extract image captions
            for img in page.get("images", []):
                image_caption = json.loads(img.get("image_annotation", "{}"))
                all_pdf += image_caption.get("description", "") + "\n"

            # Extract OCR markdown
            all_pdf += page.get("markdown", "") + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_pdf.strip())

    print(f"OCR completed. Extracted text saved to: {output_path}")

if __name__ == "__main__":
    sample_pdf_path = "./pdf_files/anime.pdf"
    output_txt_path = "./text_document/ocr_all_pdf.txt"

    if not os.path.exists(sample_pdf_path):
        print(f"❌ File not found: {sample_pdf_path}")
    else:
        get_all_pdf(sample_pdf_path, output_path=output_txt_path)
