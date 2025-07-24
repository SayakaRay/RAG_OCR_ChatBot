import os
from pathlib import Path
from pydantic import BaseModel
from load_spilt_file import AllFileLoaderAndSplit_forSendToCountSplit

# Fix missing BaseModel import in load_spilt_file.py
if not hasattr(globals(), 'MetadataPinecone'):
    from pydantic import BaseModel
    class MetadataPinecone(BaseModel):
        project_id: str
        filename: str
        page: str
        text: str
        date_upload: str
        upload_by: str

def create_test_files():
    """สร้างไฟล์ตัวอย่างสำหรับทดสอบ โดยเน้นเฉพาะ PDF"""
    import shutil
    
    # สร้างโฟลเดอร์ test_files ถ้ายังไม่มี
    test_dir = Path("./test_files")
    test_dir.mkdir(exist_ok=True)
    
    # คัดลอก PDF จากโฟลเดอร์ pdf_files ไปยัง test_files
    pdf_source_dir = Path("./pdf_files")
    if pdf_source_dir.exists():
        pdf_files = list(pdf_source_dir.glob("*.pdf"))
        if pdf_files:
            for pdf_file in pdf_files[:2]:  # เลือกเฉพาะ 2 ไฟล์แรก
                shutil.copy(pdf_file, test_dir / pdf_file.name)
                print(f"คัดลอก {pdf_file.name} ไปยัง {test_dir}")
        else:
            print(f"⚠️ ไม่พบไฟล์ PDF ใน {pdf_source_dir}")
            # สร้างไฟล์ txt เพื่อการทดสอบ
            with open(test_dir / "dummy.txt", "w", encoding="utf-8") as f:
                f.write("นี่คือไฟล์ dummy สำหรับทดสอบเนื่องจากไม่พบไฟล์ PDF\n" * 5)
    else:
        print(f"⚠️ ไม่พบโฟลเดอร์ {pdf_source_dir}")
        # สร้างไฟล์ txt เพื่อการทดสอบ
        with open(test_dir / "dummy.txt", "w", encoding="utf-8") as f:
            f.write("นี่คือไฟล์ dummy สำหรับทดสอบเนื่องจากไม่พบโฟลเดอร์ PDF\n" * 5)
    
    print(f"✅ สร้างไฟล์ทดสอบใน {test_dir} เรียบร้อยแล้ว")
    return test_dir

def main():
    """ทดสอบฟังก์ชัน AllFileLoaderAndSplit_forSendToCountSplit"""
    # สร้างไฟล์ทดสอบ
    test_dir = create_test_files()
    
    # กำหนดค่าตัวแปร
    username = "test_user"
    project_id = "test_project_123"
    
    print("\n🔍 เริ่มทดสอบฟังก์ชัน AllFileLoaderAndSplit_forSendToCountSplit...")
    print(f"📂 ไดเร็กทอรี: {test_dir}")
    print(f"👤 ผู้ใช้: {username}")
    print(f"🔑 idโปรเจค: {project_id}\n")
    
    # เรียกใช้ฟังก์ชันที่ต้องการทดสอบ
    results = AllFileLoaderAndSplit_forSendToCountSplit(
        username=username,
        directory=str(test_dir),
        project_id=project_id
    )
    
    # แสดงผลลัพธ์
    print(f"\n✅ ได้รับผลลัพธ์ทั้งหมด {len(results)} ชิ้น\n")
    
    # แสดงตัวอย่างผลลัพธ์จากแต่ละประเภทไฟล์
    file_types = {}
    
    # จัดกลุ่มตามประเภทไฟล์
    for doc in results:
        ext = os.path.splitext(doc.filename)[1]
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(doc)
    
    # แสดงตัวอย่างจากแต่ละประเภท
    for ext, docs in file_types.items():
        print(f"\n📄 ตัวอย่างจากไฟล์ {ext} (มี {len(docs)} ชิ้น):")
        sample = docs[0]  # แสดงชิ้นแรกเป็นตัวอย่าง
        print(f"  📝 ชื่อไฟล์: {sample.filename}")
        print(f"  📄 หน้า: {sample.page}")
        print(f"  🔖 รหัสโปรเจค: {sample.project_id}")
        print(f"  👤 อัพโหลดโดย: {sample.upload_by}")
        print(f"  🕒 วันที่อัพโหลด: {sample.date_upload}")
        print(f"  📋 เนื้อหา (บางส่วน): {sample.text[:100]}...")
        
        # แสดงเนื้อหาเต็มของชิ้นแรก (สำหรับดูรายละเอียด)
        if len(docs) > 1:
            print(f"\n  🔍 ข้อมูลของชิ้นที่ 2 (เนื้อหาเต็ม):")
            print(f"  {docs[1].text}")
    
    print("\n✅ ทดสอบเสร็จสิ้น")
    
if __name__ == "__main__":
    main()
