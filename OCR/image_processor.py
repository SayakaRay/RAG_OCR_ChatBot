import base64
import os
import json
import asyncio
from pathlib import Path
from mistralai import Mistral
from OCR.config import Settings, PROJECT_ROOT, IMAGE_DIR


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
                include_image_base64=True
            )
            
            result = ocr_response.pages[0].markdown
            
            return result
            
        except Exception as e:
            print(f"OCR processing error: {e}")
            return {"error": str(e)}