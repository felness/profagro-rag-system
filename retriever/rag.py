
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import os
from .preprocess import get_chunk
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.base import Embeddings
import requests
import numpy as np
from langchain import hub
from langchain_gigachat.chat_models import GigaChat
from .preprocess import extract_documents
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

class JinaEmbedding(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed_documents(self, texts):
       
        data = {
            "model": "jina-clip-v2",
            "dimensions": 1024,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": text} for text in texts]
        }

       
        response = requests.post(self.url, headers=self.headers, json=data)

        if response.status_code == 200:
            embeddings = response.json().get("data", [])
            return [np.array(e['embedding']) for e in embeddings]  
        else:
            print(f"Ошибка запроса: {response.status_code}")
            return []
        
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
    
    
def preprocess_to_rag():
    split_documents = get_chunk()

    embeddings_model = JinaEmbedding(
        api_key=os.getenv('jina_ai_api_key')
    )

    database = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings_model
    )

    retriever = database.as_retriever(k=3, search_type='similarity')

    giga_api_key = os.getenv('GigaChat_API_key')

    llm = GigaChat(verify_ssl_certs=False, credentials=giga_api_key)

    return llm, retriever
    


def create_llm_chain():
    llm, retriever = preprocess_to_rag()
    
    template = """
    Ответь на следующий вопрос, используя информацию из контекста, если не можешь дать точный ответ - 
    скажи свои мысли, а затем добавь для более точного ответа "обратитесь к официальной документации".
    Также ты имеешь чат историю за последнии 3 запроса.

    Контекст: {context}


    Чат История: {chat_history}



    Вопрос: {query}
    """

    prompt_template = ChatPromptTemplate.from_template(template=template)

    llm_chain = ({'context' : retriever | extract_documents, 'query' : RunnablePassthrough()} 
    | prompt_template 
    | llm 
    )

    return llm_chain
    






