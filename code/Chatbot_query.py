
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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseSequentialChain, SQLDatabaseChain

# FREE DATABASE ONLINE
# Host: sql8.freesqldatabase.com
# Database name: sql8709378
# Database user: sql8709378
# Database password: pTZ9fzZ2Uk
# Port number: 3306

# PINECONE_API_KEY = 'dd947cd1-d0b1-481c-8c93-5d2d9a178602' orazio
# Streaming Handler

print(load_dotenv(find_dotenv('info.env')))
os.getenv('OPENAI_API_KEY')
os.getenv('PINECONE_API_KEY')
# retrive Pinecone vectorDB
vector = PineconeVectorStore(embedding=OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),  index_name=os.getenv('PINECONE_INDEX_NAME'))

# set  searchtype as :
#   The Maximal Marginal Relevance (MMR) criterion strives to reduce redundancy while maintaining query relevance in re-ranking retrieved documents - Source

retriever = vector.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# Create tools
# set the retriver tool provinding a name and description about what knowledge will be searched  

# used to retrive info using rag for foglietti illustrativi
retriever_tool = create_retriever_tool(
    retriever,
    "AI-pharmacist",
    "Helps human interlocutors in finding the best medication that can counteract their symptoms.",
)

# Search tool
# it's used to retrive info over internet
# search = TavilySearchResults(max_results=3)
# tools = [search, retriever_tool]

# ADDING MYSQL DATABASE TO RUN STATS
# if you are using MySQL
# MySQL connection parameters
host = 'localhost'  # Or your MySQL server IP address
port = '3307'  # Default MySQL port
user = 'root'  # MySQL username
password = ''  # MySQL password
database = 'farmaci'  # MySQL database name
mysql_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'

db = SQLDatabase.from_uri(mysql_uri)


tools = [retriever_tool]
# Initialize the model
model = ChatOpenAI(model="gpt-4o", streaming=True)

db_chain = SQLDatabaseChain.from_llm(llm=model, db=db, verbose=True,
                                     return_intermediate_steps=True, top_k=1)
question = "average of tachipirina sold?"
response = db_chain(question) 
print(response)

'''
# 'SELECT COUNT(*) AS TotalAlbums\nFROM Album;'

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


if prompt := st.chat_input("What is up?", key="first_question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        response = db_chain(prompt) 
        #rint(result)

    st.session_state.messages.append({"role": "assistant", "content": response})
#    st.chat_message("assistant").markdown(response)
'''