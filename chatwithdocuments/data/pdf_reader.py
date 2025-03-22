# pdf_reader.py
import PyPDF2
from typing import List

class PDFReader:
    def __init__(self):
        self.page_list = []
        
    def read_pdf(self, file_path: str) -> List[str]:
        """
        Read PDF content and return list of pages
        Each element in the list is the text content of a page
        """
        try:
            # Open and read the PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                self.page_list = []
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:  # Only add non-empty pages
                        self.page_list.append(text.strip())
                
                return self.page_list
                
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")