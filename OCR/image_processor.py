import base64
import os
import json
import asyncio
from pathlib import Path
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from OCR.config import Settings, PROJECT_ROOT, IMAGE_DIR


from enum import Enum
from pydantic import BaseModel, Field

class ImageType(str, Enum):
    GRAPH = "graph"; TEXT = "text"; TABLE = "table"; IMAGE = "image"

class ImageAnnotation(BaseModel):
    image_type : ImageType = Field(..., description="graph / text / ...")
    description: str

class DocumentAnnotation(BaseModel):
    language: str
    summary : str
    authors : list[str]


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = Path(image_path) if not isinstance(image_path, Path) else image_path
        self.api_key = Settings.MISTRAL_KEY
        self.client = Mistral(api_key=self.api_key)
        
    def encode_image(self):
        try:
            with open(self.image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {self.image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    async def run(self):
        base64_image = self.encode_image()
        if base64_image is None:
            return {"error": f"Could not process image: {self.image_path}"}
            
        try:
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                bbox_annotation_format=response_format_from_pydantic_model(ImageAnnotation),
                document_annotation_format=response_format_from_pydantic_model(DocumentAnnotation),
                include_image_base64=True
            )
            
            # Convert to same format as PDFProcessor - return as list of dict
            data = json.loads(ocr_response.model_dump_json())
            return [data]
            
        except Exception as e:
            print(f"OCR processing error: {e}")
            return {"error": str(e)}