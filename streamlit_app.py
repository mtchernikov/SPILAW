import streamlit as st
st.set_page_config(layout="wide")

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd

# Set OpenAI API client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
OPENAI_MODEL = "gpt-4o"

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = []

st.title("📄 SPI Derivation from Uploaded Traffic Law (Autonomous Driving)")

# Step 1: Upload document
uploaded_file = st.file_uploader("Upload traffic law or regulation (.txt)", type=["txt"])
if uploaded_file:
    st.success("Document uploaded successfully.")
    raw_text = uploaded_file.read().decode("utf-8")

    # Step 2: Embed for retrieval
    with st.spinner("Embedding and indexing document..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = [Document(page_content=raw_text)]
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

    # Step 3: Derive requirements and SPI
    st.markdown("### 🔍 Query to extract legal requirements & derive SPIs")
    query = st.text_input("Query the document", value="List requirements for stopping, yielding and pedestrian safety.")
    if st.button("Run Extraction and Derivation"):
        with st.spinner("Retrieving and deriving..."):
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents(query)
            combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"""
You are an expert in traffic law and safety performance indicators (SPIs) for autonomous driving.

Given the legal text below, extract relevant legal requirements (only those that apply to **vehicle behavior**, not general public behavior) and for each:

1. Identify its **legal section** (e.g., §22101 or article number).
2. Summarize the **requirement** precisely.
3. Analyze the **ambiguity** (e.g. undefined conditions like "yield to pedestrians").
4. Propose a **recommendation** (e.g. refer to another section, define thresholds).
5. Derive a measurable **SPI** characterizing vehicle/system behavior (e.g., "time-to-yield < 2s when pedestrian detected").

Return as a markdown table with the following 5 columns:

| CVC Section | Derived Requirement | Ambiguity / Open Elements | Recommendation / Reference | Safety Performance Indicator (SPI) |

Text to analyze:
{combined_text}
"""

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a legal analyst and automotive safety engineer."},
                    {"role": "user", "content": prompt}
                ]
            )
            md_table = response.choices[0].message.content.strip()

            st.markdown("### ✅ Extracted Requirements and SPIs")
            st.markdown(md_table)

            # Parse markdown to table
            lines = md_table.splitlines()
            parsed_rows = []
            for line in lines:
                if "|" in line and not line.strip().startswith("|---"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 6:
                        parsed_rows.append({
                            "CVC Section": parts[1],
                            "Derived Requirement": parts[2],
                            "Ambiguity / Open Elements": parts[3],
                            "Recommendation / Reference": parts[4],
                            "SPI": parts[5],
                        })

            if parsed_rows:
                st.session_state["results"].extend(parsed_rows)
                st.success("✅ Parsed table successfully.")

# Step 4: Display and export
if st.session_state["results"]:
    st.markdown("### 📊 Final SPI Table")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export as CSV", data=csv, file_name="spi_table.csv", mime="text/csv")
