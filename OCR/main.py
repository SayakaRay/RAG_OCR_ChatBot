# main.py (ฉบับปรับปรุง)
import asyncio
from pathlib import Path
import traceback

# ลบ import ที่ไม่จำเป็นออกไป
# from OCR.image_manager import ImageManager
# from OCR.storage import S3Storage
# from OCR.pdf_processor import PDFProcessor
# from OCR.image_processor import ImageProcessor

# เพิ่ม Processor ใหม่ และจัดการเรื่อง Path
from OCR.powerpoint_processor import PowerPointProcessor
from OCR.typhoon_processor import TyphoonProcessor


async def pipeline(FILE_PATH, filetype, username, project_id):
    """
    ไปป์ไลน์ที่ปรับปรุงใหม่เพื่อใช้ TyphoonProcessor
    """
    try:
        file_path_obj = Path(FILE_PATH)
        processor = None

        # 1. เลือก Processor
        # สำหรับ PPTX เรายังคงต้องแปลงเป็น PDF ก่อน
        if filetype == "pptx":
            print("🔄 กำลังแปลงไฟล์ PowerPoint เป็น PDF...")
            pptx_processor = PowerPointProcessor(file_path_obj)
            # เราต้องรัน co-routine ของ pptx_processor เพื่อให้ได้ path ของ pdf ชั่วคราว
            # ส่วนนี้อาจจะต้องปรับโครงสร้างของ PowerPointProcessor เล็กน้อยเพื่อให้คืนค่า path
            # หรือรันมันเพื่อสร้างไฟล์ PDF ก่อน แล้วค่อยส่ง path นั้นให้ TyphoonProcessor
            # เพื่อความง่าย เราจะสมมติว่ามีฟังก์ชันแปลงไฟล์แยก
            # หมายเหตุ: การแปลงไฟล์ pptx ยังคงใช้ processor เดิม แต่ OCR จะใช้ตัวใหม่
            # (ส่วนนี้ซับซ้อน หากต้องการทำจริงอาจจะต้องปรับโค้ด pptx processor เพิ่มเติม)
            # **เพื่อความง่ายในตอนนี้ เราจะรองรับแค่ PDF และ รูปภาพก่อน**
            print("การประมวลผล PPTX โดยตรงกับ Typhoon ยังไม่รองรับในตัวอย่างนี้")
            return
        
        elif filetype in ["pdf", "jpg", "jpeg", "png"]:
            processor = TyphoonProcessor(file_path_obj)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

        # 2. ทำ OCR ด้วย Processor ใหม่
        if processor:
            print(f"🚀 Starting OCR with TyphoonProcessor for {FILE_PATH}...")
            ocr_results = await processor.run()
            
            if not ocr_results or ocr_results[0].get("error"):
                 print(f"❌ OCR processing failed: {ocr_results[0].get('error', 'Unknown error')}")
                 return
            print("✅ OCR processing complete.")
        else:
            print("❌ ไม่ได้เลือก Processor ที่เหมาะสม")
            return

        # 3. รวบรวมผลลัพธ์ Markdown
        # ไม่มีการประมวลผลรูปภาพอีกต่อไป เพราะ Typhoon จัดการในตัวแล้ว
        full_md_content = []
        for resp in ocr_results:
            for page in resp.get("pages", []):
                md = page.get("markdown", "")
                full_md_content.append(md)

        # 4. บันทึก Markdown ลงไฟล์
        # ใช้ Path เดิมตามที่คุณต้องการ
        output_path = Path(__file__).parent.parent / "text_document" / "ocr_output_typhoon.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ใช้ตัวคั่นที่ชัดเจนระหว่างหน้า
        output_path.write_text("\n\n---\n\n".join(full_md_content), encoding="utf-8")
        print(f"✅ Markdown content saved to: {output_path}")

        # 5. ส่วนของ ImageManager และ S3 ถูกลบออก
        # เนื่องจาก Typhoon OCR ไม่ได้คืนค่ารูปภาพออกมา
        print("✅ Pipeline finished successfully.")


    except Exception as e:
        print(f"❌ An unexpected error occurred in the pipeline: {e}")
        traceback.print_exc()

# ตัวอย่างการเรียกใช้
# if __name__ == "__main__":
#     # ต้องกำหนดค่าเหล่านี้เพื่อทดสอบ
#     TEST_FILE_PATH = "path/to/your/file.pdf" 
#     TEST_FILE_TYPE = "pdf" # หรือ "jpg"
#     TEST_USERNAME = "test_user"
#     TEST_PROJECT_ID = "project_123"
#     asyncio.run(pipeline(TEST_FILE_PATH, TEST_FILE_TYPE, TEST_USERNAME, TEST_PROJECT_ID))