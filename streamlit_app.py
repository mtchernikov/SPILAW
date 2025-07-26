import streamlit as st
st.set_page_config(layout="wide")
import os
import openai
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as OpenAILLM

# Set API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session
if "results" not in st.session_state:
    st.session_state["results"] = []

st.title("SPI Derivation from Uploaded Traffic Law (CVC)")

# Step 1: Upload CVC text file
uploaded_file = st.file_uploader("Upload traffic law .txt file", type=["txt"])
if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    st.success("Document uploaded successfully.")

    with st.spinner("Building knowledge base..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [Document(page_content=raw_text)]
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

    # Step 2: Query interface
    st.markdown("### Query and derive CVC requirements + SPIs in one step")
    query = st.text_input("Query the document", value="List all stopping and parking related traffic law violations.")

    if st.button("Derive Requirements and SPIs"):
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAILLM(temperature=0), retriever=retriever)
        requirement_text = qa_chain.run(query)

        # Prompt LLM to derive both Requirements and SPIs in one go
        with st.spinner("Generating Requirements and SPIs..."):
            llm = OpenAILLM(temperature=0)
            full_prompt = (
                "You are a traffic safety analyst. The following are sections from the California Vehicle Code. "
                "For each section, extract the CVC section number, the requirement, and a Safety Performance Indicator (SPI) that could measure compliance. "
                "Present results in a markdown table with columns:\n"
                "1. CVC Section (e.g., §22500(a))\n"
                "2. CVC Requirement\n"
                "3. Derived SPI\n\n"
                f"Text:\n{requirement_text.strip()}"
            )
            table_result = llm.invoke(full_prompt)

        st.markdown("### Derived SPI Table")
        st.markdown(table_result)

        # Parse the markdown table
        parsed_rows = []
        for line in table_result.strip().splitlines():
            if "|" in line and not line.strip().startswith("|---"):
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 4:
                    parsed_rows.append({
                        "CVC Section": parts[1],
                        "CVC Requirement": parts[2],
                        "Derived SPI": parts[3]
                    })

        if parsed_rows:
            st.session_state["results"].extend(parsed_rows)
            st.success("Entries saved to results.")

# Step 3: Display and Export
if st.session_state["results"]:
    st.markdown("### Final SPI Table")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export SPI Table as CSV", data=csv, file_name="spi_table.csv", mime="text/csv")
