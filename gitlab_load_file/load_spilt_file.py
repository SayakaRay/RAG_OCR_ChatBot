import os
from datetime import datetime
from fixthaipdf import clean
import pytz
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

#from utils.model.schemas import MetadataPinecone
from pydantic import BaseModel

class MetadataPinecone(BaseModel):
    project_id: str
    filename: str
    page: str
    text: str
    date_upload: str
    upload_by: str
    
def AllFileLoaderAndSplit_forSendToCountSplit(username, directory, project_id, timestamp=None):
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
                loader = PyMuPDFLoader(os.path.join(directory, filename))
                documents_load = text_splitter.split_documents(loader.load())
                for doc in documents_load:
                    page = doc.metadata.get("page", "None")
                    if page is not None:
                        total_page = doc.metadata.get("total_page", None)
                        if total_page is not None:
                            page = str(page + 1) + " / " + total_page
                        elif total_page is None:
                            page = str(page + 1)
                    documents.append(MetadataPinecone(
                        filename=filename,
                        page=page,
                        text=clean(doc.page_content),
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