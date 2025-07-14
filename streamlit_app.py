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

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = []

st.set_page_config(layout="wide")
st.title("SPI Generation from California Vehicle Code (CVC)")

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
    st.markdown("### SPI Extraction")
    query = st.text_input("Ask for relevant traffic rules", value="List all stopping and parking violations.")
    if st.button("Generate SPI Table"):
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAILLM(temperature=0), retriever=retriever)
        result = qa_chain.run(query)

        # Prompt LLM to generate SPI from legal text
        st.markdown("### CVC-Based SPI Table:")
        prompt = (
            "You are a traffic safety expert. The following are legal requirements from the CVC. "
            "For each, derive a Safety Performance Indicator (SPI) that can be used to measure compliance. "
            "Show the result as a markdown table with two columns:\n"
            "1. CVC Requirement (include paragraph reference)\n"
            "2. Derived SPI\n\n"
            f"Text:\n{result.strip()}"
        )

        spi_chain = OpenAILLM(temperature=0)
        table_result = spi_chain(prompt)

        st.markdown(table_result, unsafe_allow_html=True)

        # Save results to session
        st.session_state["results"].append({
            "Extracted CVC Requirements": result.strip(),
            "Generated SPI Table": table_result.strip()
        })

# Display download option
if st.session_state["results"]:
    st.markdown("### Export All Results")

    rows = []
    for item in st.session_state["results"]:
        # Attempt to parse the markdown table into rows
        lines = item["Generated SPI Table"].splitlines()
        for line in lines:
            if "|" in line and not line.strip().startswith("|---"):
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 3:
                    rows.append({
                        "CVC Requirement": parts[1],
                        "Derived SPI": parts[2]
                    })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📤 Export SPI Table as CSV", data=csv, file_name="spi_table.csv", mime="text/csv")
