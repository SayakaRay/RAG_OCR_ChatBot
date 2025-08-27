# image_manager.py

import base64, uuid, datetime
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
import os

from OCR.config import IMAGE_DIR
from OCR.storage import S3Storage

class ImageManager:
    def __init__(self, storage: S3Storage):
        self.storage = storage
        IMAGE_DIR.mkdir(exist_ok=True, parents=True)
        for file in IMAGE_DIR.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

    @staticmethod
    def _gen_name(ext="jpg") -> str:
        t = datetime.datetime.now().strftime("%Y%m%d")
        return f"{t}_{uuid.uuid4()}"

    def upload_single_image(self, image_path: Path, username: str, project_id: str) -> list[str]:
        if not image_path.exists():
            return []
        
        # สร้างชื่อที่ไม่ซ้ำกันแต่ยังคงรักษาชื่อเดิมไว้เพื่อการอ้างอิง
        image_id = f"{Path(image_path.stem)}_{self._gen_name()}"
        self.storage.upload_file(image_path.read_bytes(), image_id, username, project_id)
        print(f"☁️ Uploaded {image_path.name} to S3 with ID: {image_id}")
        return [image_id]