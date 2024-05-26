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

load_dotenv(dotenv_path='info.env')

os.getenv('OPENAI_API_KEY')
os.getenv('PINECONE_API_KEY')

# Load and prepare documents
loader = pdf.PyPDFDirectoryLoader(
    '../pdf'
)

docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
).split_documents(docs)

vector = PineconeVectorStore.from_documents(documents, OpenAIEmbeddings(), index_name=os.getenv('PINECONE_INDEX_NAME'))

print (f"Your {len(docs)} documents have been split into {len(documents)} chunks")