import asyncio, base64, json, os, uuid, datetime
from pathlib import Path
from typing import List

from pypdf import PdfReader, PdfWriter
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from enum import Enum

from OCR.config import Settings, PROJECT_ROOT, IMAGE_DIR
from OCR.image_manager import ImageManager

class ImageType(str, Enum):
    GRAPH = "graph"; TEXT = "text"; TABLE = "table"; IMAGE = "image"

class ImageAnnotation(BaseModel):
    image_type : ImageType = Field(..., description="graph / text / ...")
    description: str

class DocumentAnnotation(BaseModel):
    language: str
    summary : str
    authors : list[str]

class PDFProcessor:
    CHUNK_SIZE = 8  # pages

    def __init__(self, pdf_path: Path):
        self.pdf_path  = Path(pdf_path)
        self.reader    = PdfReader(str(pdf_path))
        self.ocr_client = Mistral(api_key=Settings.MISTRAL_KEY)

    @staticmethod
    def _encode_pdf(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def _ocr_chunk(self, start: int, end: int):
        temp = PROJECT_ROOT / f"tmp_{start+1}_{end}.pdf"
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(self.reader.pages[i])
        with open(temp, "wb") as f: writer.write(f)

        b64 = self._encode_pdf(temp)
        temp.unlink(missing_ok=True)

        def call():
            resp = self.ocr_client.ocr.process(
                model="mistral-ocr-latest",
                pages=list(range(end-start)),
                document={"type":"document_url",
                          "document_url":f"data:application/pdf;base64,{b64}"},
                bbox_annotation_format=response_format_from_pydantic_model(ImageAnnotation),
                document_annotation_format=response_format_from_pydantic_model(DocumentAnnotation),
                include_image_base64=True
            )
            data = json.loads(resp.model_dump_json())
            data["meta"] = {"start_page": start+1, "end_page": end}
            return data

        return await asyncio.to_thread(call)

    async def run(self) -> list[dict]:
        tasks = [self._ocr_chunk(i, min(i+self.CHUNK_SIZE, len(self.reader.pages)))
                 for i in range(0, len(self.reader.pages), self.CHUNK_SIZE)]
        return [r for r in await asyncio.gather(*tasks) if r]
