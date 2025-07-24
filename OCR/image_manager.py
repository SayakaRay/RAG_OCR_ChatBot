import base64, uuid, datetime
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from OCR.config import IMAGE_DIR
from OCR.storage import S3Storage

class ImageManager:
    def __init__(self, storage: S3Storage):
        self.storage = storage
        IMAGE_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _gen_name(ext="jpg") -> str:
        t = datetime.datetime.now().strftime("%Y%m%d")
        return f"{t}_{uuid.uuid4()}.{ext}"

    def save_local(self, base64_str: str) -> Path:
        data = base64.b64decode(base64_str.split(",")[1])
        name = self._gen_name()
        path = IMAGE_DIR / name
        with open(path, "wb") as f: f.write(data)
        return path

    def upload_folder(self, project_id: str) -> list[str]:
        ids = []
        for p in tqdm(IMAGE_DIR.glob("*.jpg"), desc="Upload to S3"):
            image_id = p.stem
            self.storage.upload_file(p.read_bytes(), image_id, project_id)
            ids.append(image_id)
        return ids
