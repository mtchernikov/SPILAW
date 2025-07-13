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

# Set API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define categories
categories = [
    "Functional Safety Requirement",
    "Operational Design Domain (ODD)",
    "Human-Machine Interaction",
    "Perception and Sensor Compliance",
    "Actuation and Control",
    "Fallback and Fail-Operational",
    "Post-Market Monitoring",
    "Cybersecurity Interface",
    "Legal/Traffic Compliance",
    "Not Categorized"
]

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = []

st.title("SPI Extraction and Categorization from CVC Upload")

# Step 1: Upload document
uploaded_file = st.file_uploader("Upload a CVC or traffic law .txt file", type=["txt"])
if uploaded_file:
    st.success("Document uploaded successfully.")
    raw_text = uploaded_file.read().decode("utf-8")

    # Step 2: Build Vector DB
    with st.spinner("Processing and embedding document..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [Document(page_content=raw_text)]
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

    # Step 3: Run query
    st.markdown("### Extract SPI-relevant legal requirements:")
    query = st.text_input("Query the document", value="List all stopping and parking violations.")
    if st.button("Extract Requirements"):
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAILLM(temperature=0), retriever=retriever)
        result = qa_chain.run(query)

        st.markdown("### Retrieved Legal Requirement:")
        st.code(result, language="text")

        st.markdown("### Categorize the extracted requirement:")
        selected_category = st.selectbox("Choose category", categories)
        if st.button("Confirm Category"):
            st.session_state["results"].append({
                "Requirement": result.strip(),
                "Category": selected_category
            })
            st.success(f"Requirement categorized as: **{selected_category}**")

# Display results table and export
if st.session_state["results"]:
    st.markdown("### Categorized Results Table")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export as CSV", data=csv, file_name="categorized_spi_requirements.csv", mime="text/csv")
