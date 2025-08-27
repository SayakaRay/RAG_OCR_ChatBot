# check_env.py
import os
from dotenv import load_dotenv

print("--- Running .env check ---")

# verbose=True จะบอกว่ามันหาไฟล์ .env เจอที่ไหน
found_dotenv = load_dotenv(verbose=True)

if found_dotenv:
    print("✅ Found .env file.")
else:
    print("❌ WARNING: Did not find a .env file in the current directory or parent directories.")
    print(f"Current Directory: {os.getcwd()}")

api_key = os.getenv("TYPHOON_API_KEY")

if api_key:
    print(f"✅ Successfully loaded TYPHOON_API_KEY.")
    # แสดง Key 4 ตัวท้ายเพื่อยืนยัน
    print(f"   Key starts with: {api_key[:5]}...{api_key[-4:]}")
else:
    print("❌ FAILED to load TYPHOON_API_KEY from environment.")