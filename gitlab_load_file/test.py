import os
from datetime import datetime
from load_spilt_file import AllFileLoaderAndSplit_forSendToCountSplit  # Replace with actual module name
from pathlib import Path
import asyncio

# Define project root and input directory
project_root = Path(__file__).resolve().parent.parent
input_dir = project_root / "gitlab_load_file" / "input_file"

# Setup test values
username = "test_user"
project_id = "test_project"
timestamp = datetime.now()

# Run the loader function
documents = asyncio.run(AllFileLoaderAndSplit_forSendToCountSplit(
    username=username,
    directory=str(input_dir),
    project_id=project_id,
    timestamp=timestamp
))

# Print outputs
print(f"\n--- Loaded {len(documents)} chunks ---\n")
for i, doc in enumerate(documents):
    print(f"Chunk #{i+1}")
    print(f"Filename: {doc.filename}")
    print(f"Page: {doc.page}")
    print(f"Text: {doc.text[:300]}...")
    print(f"Uploaded by: {doc.upload_by}")
    print(f"Date: {doc.date_upload}")
    print(f"Project ID: {doc.project_id}")
    print("------\n")
