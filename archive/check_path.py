# check_path.py
import sys
import os

print("--- Python Executable ---")
print("นี่คือ Python ที่กำลังถูกใช้งาน:")
print(sys.executable)
print("-" * 25)

print("\n--- Python Search Path (sys.path) ---")
print("Python กำลังค้นหาไลบรารีจากโฟลเดอร์เหล่านี้ตามลำดับ:")
for p in sys.path:
    print(p)
print("-" * 25)

print("\n--- Attempting to find 'surya' module ---")
try:
    import surya
    print("\n[SUCCESS] ✅ พบโมดูล 'surya' ที่ไฟล์:")
    # __file__ จะบอกเราว่า __init__.py ของ surya มาจากไหน
    print(surya.__file__) 
except ImportError as e:
    print(f"\n[FAIL] ❌ ไม่สามารถ import 'surya' ได้: {e}")
except Exception as e:
    print(f"\n[FAIL] ❌ เกิด Error อื่นๆ: {e}")