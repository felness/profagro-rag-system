import streamlit as st 
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langchain_core.output_parsers import StrOutputParser
import os
import sys  
# Добавление пути к родительскому каталогу в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retriever.rag import create_llm_chain



MAX_HISTORY_LENGTH = 3



if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
st.set_page_config(page_title="Streaming Bot", page_icon='🤖')

st.title('assistant')

# get response
def get_response(query, chat_history):
    
    # limited_history = chat_history[-MAX_HISTORY_LENGTH:]
    
    chain = create_llm_chain()
    
    # history_str = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in limited_history])
    
    return chain.invoke(query).content
    
# conversation
for message in  st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    else :
        with st.chat_message('AI'):
            st.markdown(message.content)
            
            
#user input
user_query = st.chat_input('Your message')

if user_query is not None and user_query!="":
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message('Human'):
        st.markdown(user_query)
    
    with st.chat_message('AI'):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)
    
    st.session_state.chat_history.append(AIMessage(ai_response))