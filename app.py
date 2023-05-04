import re
import time
from io import BytesIO
from typing import Any, Dict, List

# Modules to Import
import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

@st.cache_resource
def construct_index(directory_path):
    max_input_size = 16384
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


@st.cache_resource   
def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot-NCERT")
st.title("🤖 Personalized NCERT Bot 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Chat with your NCERT Books 📜       
        ----
        """
)

index = construct_index("docs")

query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything",
            )

if query:
    with st.spinner(
        "Generating Answer to your Query "):
        res = chatbot(query)
        st.info(res, icon="🤖")


