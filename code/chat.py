# %%
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

load_dotenv(dotenv_path='info.env')

host = 'sql8.freesqldatabase.com'  # Or your MySQL server IP address
port = '3306'  # Default MySQL port
user = 'sql8709427'  # MySQL username
password = 'TdVxdLEFft'  # MySQL password
database = 'sql8709427'  # MySQL database name
mysql_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'

class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

db_engine=create_engine(mysql_uri)
db=SQLDatabase(db_engine)

# retrive Pinecone vectorDB
vector = PineconeVectorStore(embedding=OpenAIEmbeddings(),  index_name=os.getenv('PINECONE_INDEX_NAME'))
retriever = vector.as_retriever(search_type="mmr", search_kwargs={"k": 6})

model=ChatOpenAI(temperature=0.0,model="gpt-4o")
sql_toolkit=SQLDatabaseToolkit(db=db,llm=model)
sql_toolkit.get_tools()

# used to retrive info using rag for foglietti illustrativi
retriever_tool = create_retriever_tool(
    retriever,
    "AI-pharmacist",
    "Helps human interlocutors in finding the best medication that can counteract their symptoms.",
)

sql_tool = sql_toolkit.get_tools()

tools = [retriever_tool] + sql_tool
# %%

prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a very intelligent system. Your first aim is to identify the correct type of agent to use to answer to the user question. \
                    Once identified, be that agent! The questions could be either or a mysql query or informative query. \
                    If you recognize a mysql query, identify relevant questions from user and converting into sql queries to generate correct answer. \
                    Please use the below context to write the microsoft sql queries , dont use mysql queries. \
                        MySQL-context: you must query against the connected database, it has total 1 tables , this is transactions transaction table has \
                        ['Date', 'Codice AIC', 'Quantity', 'Principio Attivo', 'Descrizione Gruppo', 'Denominazione e Confezione', 'Prezzo al pubblico', \
                        'Titolare AIC', 'Codice Gruppo Equivalenza']. \
                    On the other side, if you recognize a informative, read the pdfs and use the retrivel tool to helps human interlocutors in finding the best medication that can counteract their symptoms."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Streamlit app
st.set_page_config(page_title="Pharmacist Assistant", page_icon=":-)")
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
#     st.divider()
#     st.markdown("Chiedimi i contatti della segreteria!")

if prompt := st.chat_input("What is up?", key="first_question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        stream_handler = StreamHandler(st.empty())
        # Execute the agent with chat history
        result = agent_executor(
            {
                "input": prompt,
                "chat_history": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            },
            callbacks=[stream_handler],
        )
#        print(result)
        response = result.get("output")

    st.session_state.messages.append({"role": "assistant", "content": response})
#    st.chat_message("assistant").markdown(response)