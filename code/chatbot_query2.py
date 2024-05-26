from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import SQLDatabase

print(load_dotenv(find_dotenv('info.env')))

host = 'localhost'  # Or your MySQL server IP address
port = '3307'  # Default MySQL port
user = 'root'  # MySQL username
password = ''  # MySQL password
database = 'farmaci'  # MySQL database name
mysql_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'

db_engine=create_engine(mysql_uri)
db=SQLDatabase(db_engine)


llm=ChatOpenAI(temperature=0.0,model="gpt-4o")
sql_toolkit=SQLDatabaseToolkit(db=db,llm=llm)
sql_toolkit.get_tools()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        you are a very intelligent AI assitasnt who is expert in identifying relevant questions from user and converting into sql queriesa to generate correcrt answer.
        Please use the below context to write the microsoft sql queries , dont use mysql queries.
       context:
       you must query against the connected database, it has total 1 tables , this is transactions 
       transaction table has ['Date', 'Codice AIC', 'Quantity', 'Principio Attivo',
       'Descrizione Gruppo', 'Denominazione e Confezione',
       'Prezzo al pubblico ', 'Titolare AIC', 'Codice Gruppo Equivalenza']

       When the user specify a name of drugs search always the names that contains that name
       
   
        """
        ),
        ("user","{question}\ ai: ")
    ]
)
agent=create_sql_agent(llm=llm,toolkit=sql_toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,max_execution_time=100,max_iterations=50)



# Streamlit app
st.set_page_config(page_title="Pharmacist Assistant v2", page_icon=":-)")
st.header("Your AI-based pharmacist")
st.write(
    """Hi. I am an agent powered by Neodata.
I will be your virtual assistant to help you with your experience within the student portal. 
Ask me anything about your courses or academic career"""
)
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/neodatagroup/hackathon_RAG/tree/main)"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("Farmaci acquisiti")
    st.markdown("Tachipirina")
    st.markdown("Nurofen")
    st.markdown("Toradol")


if question := st.chat_input("What is up?", key="first_question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        
       result = agent.run(prompt.format_prompt(question=question))
       #response = result.get("output")
       #print(result)

    #st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").markdown(result)