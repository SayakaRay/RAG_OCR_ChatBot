# typhoon_processor.py
import asyncio
from pathlib import Path
from pypdf import PdfReader
import sys

# เพิ่ม Path ของโปรเจกต์หลักเพื่อให้สามารถ import typhoon_tranformers_app_API ได้
# สมมติว่าไฟล์ typhoon_tranformers_app_API.py อยู่ในโฟลเดอร์เดียวกับโฟลเดอร์ OCR
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typhoon_tranformers_app_API import process_pdf_via_api, image_to_pdf

class TyphoonProcessor:
    """
    Processor ที่ห่อหุ้ม Typhoon OCR API เพื่อให้ทำงานร่วมกับ pipeline หลักได้
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        # ตรวจสอบว่าไฟล์เป็นรูปภาพหรือไม่
        self.is_image = file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']

    async def run(self) -> list[dict]:
        """
        รัน OCR process โดยเรียกใช้ process_pdf_via_api สำหรับทุกหน้า
        และจัดรูปแบบผลลัพธ์ให้ตรงตามที่ main.py ต้องการ
        """
        if self.is_image:
            # ถ้าเป็นรูปภาพ ให้มีหน้าเดียว
            num_pages = 1
            pdf_path_to_process = self.file_path
        else:
            # ถ้าเป็น PDF ให้นับจำนวนหน้า
            try:
                reader = PdfReader(self.file_path)
                num_pages = len(reader.pages)
                pdf_path_to_process = self.file_path
            except Exception as e:
                print(f"❌ ไม่สามารถอ่านไฟล์ PDF: {e}")
                return [{"error": str(e)}]

        all_pages_data = []
        print(f"📄 พบเอกสารทั้งหมด {num_pages} หน้า จะทำการประมวลผลทีละหน้า...")

        # วนลูปเพื่อประมวลผลทุกหน้า
        for i in range(1, num_pages + 1):
            page_num = i
            print(f"⚙️ กำลังประมวลผลหน้า {page_num}/{num_pages}...")
            
            # เนื่องจาก process_pdf_via_api เป็นฟังก์ชัน synchronous (blocking)
            # เราจึงต้องใช้ asyncio.to_thread เพื่อรันใน thread แยก ไม่ให้ block event loop
            def sync_process():
                # Task type สามารถเปลี่ยนได้ตามต้องการ 'structure' หรือ 'default'
                _, markdown_result = process_pdf_via_api(
                    pdf_or_image_file=str(pdf_path_to_process),
                    page_num=page_num,
                    task_type="structure" 
                )
                return markdown_result

            try:
                markdown_content = await asyncio.to_thread(sync_process)
                
                if markdown_content:
                    # สร้างโครงสร้างข้อมูลที่ main.py คาดหวัง
                    # Typhoon API ไม่คืนค่ารูปภาพแยกมาให้ ดังนั้น "images" จะเป็น list ว่าง
                    page_data = {
                        "markdown": markdown_content,
                        "images": [] 
                    }
                    all_pages_data.append(page_data)
                else:
                    print(f"⚠️ ไม่ได้รับผลลัพธ์จากหน้า {page_num}")

            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในการประมวลผลหน้า {page_num}: {e}")
                continue
        
        # ห่อหุ้มผลลัพธ์ทั้งหมดในโครงสร้างที่ pipeline คาดหวัง
        return [{"pages": all_pages_data}]