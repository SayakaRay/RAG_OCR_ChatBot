# OCR/typhoon_processor.py

import asyncio
from pathlib import Path
from pypdf import PdfReader
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typhoon_tranformers_app_API import process_pdf_via_api

class TyphoonProcessor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
    async def run(self) -> list[dict]:
        try:
            reader = PdfReader(self.file_path)
            num_pages = len(reader.pages)
        except Exception as e:
            print(f"❌ ไม่สามารถอ่านไฟล์ PDF เพื่อนับจำนวนหน้า: {e}")
            return [{"error": str(e)}]
        all_pages_data = []
        print(f"📄 พบเอกสารทั้งหมด {num_pages} หน้า จะทำการประมวลผลทีละหน้า...")
        for i in range(1, num_pages + 1):
            def sync_process():
                _, markdown_result = process_pdf_via_api(pdf_path=str(self.file_path), page_num=i)
                return markdown_result
            try:
                markdown_content = await asyncio.to_thread(sync_process)
                if markdown_content is not None:
                    all_pages_data.append({"markdown": markdown_content, "images": []})
                else:
                    all_pages_data.append({"markdown": f"Error processing page {i}.", "images": []})
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในการเรียก process_pdf_via_api สำหรับหน้า {i}: {e}")
                continue
        return [{"pages": all_pages_data}]