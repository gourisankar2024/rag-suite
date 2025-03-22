# document_loader.py
import os
from typing import Optional

class DocumentLoader:
    def __init__(self):
        self.uploaded_file = None
        
    def load_file(self, file_path: str) -> Optional[str]:
        """
        Load the uploaded PDF file and validate it
        Returns the file path if valid, None otherwise
        """
        if not file_path:
            return None
            
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are supported")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError("File does not exist")
            
        self.uploaded_file = file_path
        return file_path