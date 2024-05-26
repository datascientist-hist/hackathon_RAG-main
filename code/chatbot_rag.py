
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



# PINECONE_API_KEY = 'dd947cd1-d0b1-481c-8c93-5d2d9a178602' orazio
# Streaming Handler
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


load_dotenv(dotenv_path='info.env')

# retrive Pinecone vectorDB
vector = PineconeVectorStore(embedding=OpenAIEmbeddings(),  index_name=os.getenv('PINECONE_INDEX_NAME'))

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


tools = [retriever_tool]
# Initialize the model
model = ChatOpenAI(model="gpt-4o", streaming=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a pharmacist. Answer always in the language of the question. If you don't know the answer, just say 'I don't know', don't invent the answer!",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent that decide what tools to call
# Tool calling allows a model to detect when one or more tools should be called and respond with the inputs that should be passed to those tools.
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
        print(result)
        response = result.get("output")

    st.session_state.messages.append({"role": "assistant", "content": response})
#    st.chat_message("assistant").markdown(response)
