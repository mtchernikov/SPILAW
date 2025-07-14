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
uploaded_file = st.file_uploader("Upload a CVC or traffic law .txt file", type=["txt"])
if uploaded_file:
    st.success("Document uploaded successfully.")
    raw_text = uploaded_file.read().decode("utf-8")

    # Step 2: Embed document and build FAISS index
    with st.spinner("Processing and embedding document..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [Document(page_content=raw_text)]
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

    # Step 3: Query interface
    st.markdown("### Extract SPI-relevant legal requirements:")
    query = st.text_input("Query the document", value="List all stopping and parking violations.")
    if st.button("Extract Requirements"):
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAILLM(temperature=0), retriever=retriever)
        result = qa_chain.run(query)

        st.markdown("### Retrieved Legal Requirement:")
        st.code(result, language="text")

        # Step 4: SPI derivation prompt
        st.markdown("### Derive SPIs for each CVC requirement:")
        with st.spinner("Deriving SPI..."):
            llm = OpenAILLM(temperature=0)
            prompt = (
                "You are a traffic safety expert. The following are legal requirements from the CVC. "
                "For each, derive a Safety Performance Indicator (SPI) that can be used to measure compliance. "
                "Show the result as a markdown table with three columns:\n"
                "1. CVC Section (e.g. §22500(a))\n"
                "2. CVC Requirement\n"
                "3. Derived SPI\n\n"
                f"Text:\n{result.strip()}"
            )
            table_result = llm.invoke(prompt)

            st.markdown("### SPI Table")
            st.markdown(table_result)

            # Parse markdown table into structured format
            lines = table_result.strip().splitlines()
            parsed_rows = []

            for line in lines:
                if "|" in line and not line.strip().startswith("|---"):
                    parts = [part.strip() for part in line.split("|")]
                    if len(parts) >= 4:
                        section = parts[1]
                        requirement = parts[2]
                        spi = parts[3]
                        parsed_rows.append({
                            "CVC Section": section,
                            "CVC Requirement": requirement,
                            "Derived SPI": spi
                        })

            if parsed_rows:
                st.session_state["results"].extend(parsed_rows)
                st.success("SPIs extracted and saved to session.")

# Step 5: Display and export results
if st.session_state["results"]:
    st.markdown("### Final Table of SPIs")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📤 Export SPI Table as CSV", data=csv, file_name="spi_table.csv", mime="text/csv")
