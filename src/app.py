import streamlit as st 
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
api_key = os.getenv('GigaChat_API_key')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
st.set_page_config(page_title="Streaming Bot", page_icon='🤖')

st.title('ProfAgro Bot')
# get response
def get_response(query, chat_history):
    template = """
    Ты ассистент. Ответь на вопрос представленный ниже.
    
    Chat_history:{chat_history}
    
    User Query: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    llm = GigaChat(verify_ssl_certs=False, credentials=api_key)
    
    chain = (prompt | llm | StrOutputParser())
    return chain.invoke({
        'chat_history' : chat_history,
        'question' : query
    })
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