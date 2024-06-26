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
import glob
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import SQLDatabase

# %% load_dotenv(dotenv_path='info.env')
load_dotenv(find_dotenv('info.env'))

# %% set db
host = 'sql8.freesqldatabase.com'  # Or your MySQL server IP address
port = '3306'  # Default MySQL port
user = 'sql8709427'  # MySQL username
password = 'TdVxdLEFft'  # MySQL password
database = 'sql8709427'  # MySQL database name
mysql_uri = f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}'

db_engine=create_engine(mysql_uri)
db=SQLDatabase(db_engine)

# %% define agent
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

# retrive Pinecone vectorDB
vector = PineconeVectorStore(embedding=OpenAIEmbeddings(),  index_name=os.getenv('PINECONE_INDEX_NAME'))
retriever = vector.as_retriever(search_type="mmr", search_kwargs={"k": 6})

model=ChatOpenAI(temperature=0.0,model="gpt-3.5-turbo")
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

prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a very intelligent system. Your first aim is to identify the correct tool to use to answer to the user question. \
                    Once identified, use it! The questions could be either or a mysql query or informative query. \
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
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

def find_file_in_folder(filename):
    folder_path = os.path.abspath('../pdf')
    # Search for the file in the specified folder and its subfolders
    search_pattern = os.path.join(folder_path, filename)
    found_files = glob.glob(search_pattern, recursive=True)
    if found_files:
        # Return the complete paths of the found files
        return found_files
    else:
        return None

def source_files(documents):
    # List to store extracted metadata
    metadata_list = []
    # Extract metadata from each document
    for document in documents:
        metadata = {
            'source': document.metadata.get('source'),
            'page': document.metadata.get('page')
        }
        metadata_list.append(metadata)
    return metadata_list

def format_metadata_to_markdown(metadata_list):
    markdown_string = "\n###### Sources used:\n"
    unique_path = [metadata['source'] for metadata in metadata_list]
    unique_path = set(unique_path)
    for idx, metadata in enumerate(unique_path):
        complete_path =  find_file_in_folder(metadata.split('\\')[-1])
        complete_path = complete_path[0].replace("\\", "/")
        markdown_string += f"- [file_{idx}: {metadata}](file://{complete_path})\n"
    return markdown_string

st.set_page_config(page_title="Pharmacist Assistant", page_icon=":pill:")
st.header("AI Pharmacist Assistant")

st.write(
    """
    Hi! I am your pharmacy's virtual assistant and I am here to make your stock management 
    more efficient and easier. My main task is to help you quickly find the most suitable medicines for your customers, 
    based on their symptoms and physical characteristics. In addition, I can collect and provide you 
    with useful statistics, such as identifying the best-selling drug in a certain period or 
    calculating the revenue from sales. I am here to answer all your questions and support you 
    in every aspect of your pharmacy management. 

    Rely on me to optimise your work and improve your efficiency. Enjoy!
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.image('../media/ai_pharm.png')
    st.divider()    
    st.header("Main features:")
    st.markdown("- Efficient Stock Management\n\n- Statistics Collection\n\n- Constant Support")
    st.divider()
    st.header("Authors")
    st.markdown("Orazio Pontorno  [![](https://img.shields.io/badge/-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/opontorno) [![](https://img.shields.io/badge/-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/orazio-pontorno/)")
    st.markdown("Giuseppe Pulino  [![](https://img.shields.io/badge/-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/datascientist-hist) [![](https://img.shields.io/badge/-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/giuseppe-pulino-40a9981b2/)")
    st.divider()
    st.write("[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/datascientist-hist/hackathon_RAG-main)")

if prompt := st.chat_input("What can I help you with?", key="first_question"):
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
        try:
            if (result['intermediate_steps'][0][0].tool == 'AI-pharmacist'):
                source = source_files(retriever.invoke(prompt))
                markdown_s = format_metadata_to_markdown(source)
                response += markdown_s
                st.markdown(markdown_s, unsafe_allow_html=True)
        except:
            pass
    st.session_state.messages.append({"role": "assistant", "content": response})
#    st.chat_message("assistant").markdown(response)