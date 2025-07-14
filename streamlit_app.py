import streamlit as st
import os
import openai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as OpenAILLM
import pandas as pd

# Set API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = []

st.title("SPI Derivation from Uploaded CVC Law")

# Step 1: Upload CVC or law text file
u
