import os
from datetime import datetime
from fixthaipdf import clean
import pytz
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel
from pathlib import Path
import sys
#from utils.model.schemas import MetadataPinecone
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from OCR.main import pipeline as ocr_pipeline
from chunk_text import split_text_with_langchain

class MetadataPinecone(BaseModel):
    project_id: str
    filename: str
    page: str
    text: str
    date_upload: str
    upload_by: str
    
async def AllFileLoaderAndSplit_forSendToCountSplit(username, directory, project_id, timestamp=None):
    documents = []

    # Include Timestamp so all file will have the same timestamp to metadata in db. see more: /upload
    if timestamp:
        date_upload = str(timestamp.strftime('%Y-%m-%d %H:%M:%S') + ' (Asia/Bangkok)')
    else:
        th_timezone = pytz.timezone('Asia/Bangkok')
        th_time = datetime.now(th_timezone)
        date_upload = str(th_time.strftime('%Y-%m-%d %H:%M:%S') + ' (Asia/Bangkok)')

    try:
        for filename in os.listdir(directory):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            if filename.endswith(".csv"):
                loader = CSVLoader(os.path.join(directory, filename), encoding='utf-8')
                documents_load = text_splitter.split_documents(loader.load())
                for doc in documents_load:
                    if '_SheetNameIs_' in filename:
                        filename_part = filename.split('_SheetNameIs_')
                        documents.append(MetadataPinecone(
                            filename=filename_part[0] + '.xlsx',
                            page=filename_part[1].replace('.csv', ''),
                            text=doc.page_content,
                            date_upload=date_upload,
                            upload_by=username,
                            project_id=project_id
                        ))
                    else:
                        documents.append(MetadataPinecone(
                            filename=filename,
                            page="None",
                            text=doc.page_content,
                            date_upload=date_upload,
                            upload_by=username,
                            project_id=project_id
                        ))
            elif filename.endswith(".txt"):
                loader = TextLoader(os.path.join(directory, filename), encoding='utf-8')
                documents_load = text_splitter.split_documents(loader.load())
                for doc in documents_load:
                    documents.append(MetadataPinecone(
                        filename=filename,
                        page="None",
                        text=doc.page_content,
                        date_upload=date_upload,
                        upload_by=username,
                        project_id=project_id
                    ))
            elif filename.endswith(".pdf"):
                project_root = Path(__file__).resolve().parent.parent
                text_output_path = project_root / "text_document" / "all_pdf.txt"

                await ocr_pipeline(os.path.join(directory, filename), "pdf", username, project_id)
                # OCR 
                if not text_output_path.exists():
                    print(f"OCR failed or no text found for {filename}")
                    continue

                with open(text_output_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents_load = split_text_with_langchain(text, chunk_size=1024, chunk_overlap=100)

                for i, chunk in enumerate(documents_load):
                    documents.append(MetadataPinecone(
                        filename=filename,
                        page=str(i + 1),
                        text=clean(chunk),
                        date_upload=date_upload,
                        upload_by=username,
                        project_id=project_id
                    ))
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(os.path.join(directory, filename))
                documents_load = text_splitter.split_documents(loader.load())
                for doc in documents_load:
                    documents.append(MetadataPinecone(
                        filename=filename,
                        page="None",
                        text=doc.page_content,
                        date_upload=date_upload,
                        upload_by=username,
                        project_id=project_id
                    ))
            elif filename.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(os.path.join(directory, filename))
                documents_load = text_splitter.split_documents(loader.load())
                for doc in documents_load:
                    documents.append(MetadataPinecone(
                        filename=filename,
                        page="None",
                        text=doc.page_content,
                        date_upload=date_upload,
                        upload_by=username,
                        project_id=project_id
                    ))
    except Exception as e:
        print(f"Error loading files from directory {directory}: {e}")
    finally:
        return  documents