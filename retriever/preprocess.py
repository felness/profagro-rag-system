import fitz
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import sys
import os
# Путь к родительской папке
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Путь к папке docs на уровень выше
path = os.path.join(parent_dir, 'docs')


# get pdfs_list
def get_pdf(path):
    list_with_pdf = [doc for doc in os.listdir(path) if doc.lower().endswith(".pdf")]
    return list_with_pdf

# extract text
def get_text_from_pdf():
    files = get_pdf(path)
    text = ""
    
    for file in files:
        doc = fitz.open(os.path.join(path, file))
        
        for page_num in range(2, len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    
    return text


def get_chunk():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    texts = get_text_from_pdf()
    
    split_doc = text_splitter.create_documents([texts])
    
    return split_doc

def extract_documents(document):
    return '\n\n'.join(doc.page_content for doc in document)


        
    
        
        
            
    