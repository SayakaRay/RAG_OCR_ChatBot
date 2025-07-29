import os
from pathlib import Path
from datetime import datetime
import pytz
import sys

# เพิ่ม path เพื่อให้ import โมดูลจากโฟลเดอร์หลักได้
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import โมดูลที่จำเป็น
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    print("⚠️ กรุณาติดตั้ง langchain และ pymupdf ก่อนด้วยคำสั่ง: pip install langchain langchain-community pymupdf")
    sys.exit(1)

# นิยาม MetadataPinecone class
from pydantic import BaseModel

class MetadataPinecone(BaseModel):
    project_id: str
    filename: str
    page: str
    text: str
    date_upload: str
    upload_by: str

def process_pdf(file_path, username, project_id, date_upload):
    """ประมวลผลไฟล์ PDF และสร้าง metadata"""
    print(f"📄 กำลังประมวลผล PDF: {file_path}")
    
    try:
        # โหลดและแยกเอกสาร PDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # แยกเนื้อหาเป็นชิ้นเล็กๆ
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        # สร้าง MetadataPinecone objects
        results = []
        for i, doc in enumerate(split_docs):
            metadata = MetadataPinecone(
                project_id=project_id,
                filename=os.path.basename(file_path),
                page=f"page_{doc.metadata.get('page', i+1)}",
                text=doc.page_content,
                date_upload=date_upload,
                upload_by=username
            )
            results.append(metadata)
        
        print(f"✅ ประมวลผล PDF เสร็จสิ้น: {len(results)} ชิ้น")
        return results
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการประมวลผล PDF: {e}")
        return []

def create_test_files():
    """สร้างโฟลเดอร์ทดสอบและคัดลอกไฟล์ PDF"""
    # สร้างโฟลเดอร์ test_files ถ้ายังไม่มี
    test_dir = Path("./test_files")
    test_dir.mkdir(exist_ok=True)
    
    # คัดลอก PDF จากโฟลเดอร์หลัก
    pdf_source_dir = Path("../pdf_files")
    if pdf_source_dir.exists():
        import shutil
        pdf_files = list(pdf_source_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files[:2]:  # คัดลอกเพียง 2 ไฟล์
                shutil.copy(pdf_file, test_dir / pdf_file.name)
                print(f"📋 คัดลอก {pdf_file.name} ไปยัง {test_dir}")
        else:
            print(f"⚠️ ไม่พบไฟล์ PDF ใน {pdf_source_dir}")
    else:
        print(f"⚠️ ไม่พบโฟลเดอร์ {pdf_source_dir}")
    
    return test_dir

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มการทดสอบ PDF Processor...")
    
    # สร้างไฟล์ทดสอบ
    test_dir = create_test_files()
    
    # กำหนดพารามิเตอร์
    username = "test_user"
    project_id = "test_project_pdf"
    
    # สร้าง timestamp
    th_timezone = pytz.timezone('Asia/Bangkok')
    th_time = datetime.now(th_timezone)
    date_upload = str(th_time.strftime('%Y-%m-%d %H:%M:%S') + ' (Asia/Bangkok)')
    
    # ประมวลผล PDF ทั้งหมดในโฟลเดอร์ทดสอบ
    all_results = []
    for file_path in test_dir.glob("*.pdf"):
        results = process_pdf(str(file_path), username, project_id, date_upload)
        all_results.extend(results)
    
    # แสดงผลลัพธ์
    print(f"\n✅ ประมวลผลเสร็จสิ้น: {len(all_results)} ชิ้นข้อมูล")
    
    # แสดงตัวอย่างผลลัพธ์
    if all_results:
        sample = all_results[0]
        print("\n📝 ตัวอย่างข้อมูล:")
        print(f"  📄 ชื่อไฟล์: {sample.filename}")
        print(f"  📃 หน้า: {sample.page}")
        print(f"  🔑 รหัสโปรเจค: {sample.project_id}")
        print(f"  👤 ผู้อัพโหลด: {sample.upload_by}")
        print(f"  🕒 วันที่อัพโหลด: {sample.date_upload}")
        print(f"  📋 เนื้อหา (ตัวอย่าง): {sample.text[:100]}...")
        
        # แสดงข้อมูลโดยละเอียด
        if len(all_results) > 1:
            print("\n🔍 ข้อมูลชิ้นที่ 2 (เนื้อหาเต็ม):")
            print(f"{all_results[1].text}")
    else:
        print("⚠️ ไม่มีผลลัพธ์ กรุณาตรวจสอบไฟล์ PDF ในโฟลเดอร์ทดสอบ")

if __name__ == "__main__":
    main()
