# Dockerfile
FROM python:3.12.3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# ทำ docker ignore พวก env folder ด้วย เพราะไม่อยากให้เอา env เข้าไปใน image => add ใน file .dockerignore


# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5050 # ไปแก้เป็น port ที่จะใช้

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"] # แก้เป็น port และชื่อไฟล์ที่จะใช้

# # Run app.py when the container launches
# CMD uvicorn main:app --host 0.0.0.0 --port 5050
# #--host 0.0.0.0: กำหนดให้ Uvicorn รับการเชื่อมต่อจากทุก IP address (หรือทุก interface) ภายใน container (เพื่อให้สามารถเข้าถึงจากภายนอก container ได้)
# #--port 5050: กำหนดให้ Uvicorn รันแอปพลิเคชันที่ port 5050 (อนุญาตให้เข้าถึงแอปผ่าน port 5050 ภายใน container)

# ที่สำคัญ เราจะให้ image ไปอยู่ใน app แล้วการ mount volume ก็ต้องเริ่ม set ที่ app เสมอ ตอนแจ้ง devops