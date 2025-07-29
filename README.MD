# RAG Backend & UI

This repository contains a **Retrieval-Augmented Generation (RAG)** system built with FastAPI for the backend and a simple UI frontend written in Python. It utilizes APIs such as **OpenAI**, **Mistral**, and AWS S3 for storage.

## 🔧 Requirements

* Python 3.12.11
* Install all dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1. Backend Server (FastAPI + RAG)

To run the backend RAG server locally:

```bash
uvicorn app:app --reload
```

This will start the FastAPI server at `http://127.0.0.1:8000` by default.

### 2. Frontend UI

To run the frontend interface:

```bash
python ui.py
```

This will launch the UI for interacting with the RAG system.

### 3. PDF OCR Testing App

A separate small app exists for testing the OCR functionality with the `MetadataPinecone` format. It reads PDF files and is located in the `gitlab_load_file` folder.

To run the test app:

```bash
python gitlab_load_file/test.py
```

This script uses `load_spilt_file.py` to process PDF files using your custom OCR pipeline.

## 🔐 Environment Variables

Create a `.env` file in the project root with the following content:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
AWS_ACCESS_KEY=your_aws_access_key_here
AWS_BUCKET_NAME=your_bucket_name_here
AWS_SECRET_KEY=your_aws_secret_key_here
```

These credentials are required for API access and cloud storage.

## 📂 Project Structure (Example)

```
.
├── app.py                    # FastAPI backend server
├── ui.py                     # Frontend interface
├── requirements.txt          # Python dependencies
├── .env                      # API and AWS keys (not committed)
├── gitlab_load_file/
│   ├── load_spilt_file.py    # PDF OCR processor with MetadataPinecone format
│   └── test.py               # Script to test the OCR processor
└── ...                       # Additional files/modules
```

## 📌 Notes

* Ensure Python 3.12.11 is installed and set as your environment default.
* Use `uvicorn` to serve the API backend.
* The `.env` file is critical for accessing external services and should not be committed to version control.

---

For further development, documentation, or contributions, feel free to open issues or pull requests.

---

**License**: [MIT](LICENSE) *(or your license of choice)*
