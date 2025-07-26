import streamlit as st
st.set_page_config(layout="wide")

import openai
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# -------------------------
# Config
# -------------------------
OPENAI_MODEL = "gpt-4o-mini"  # adapt to what you have
TOP_K = 8                     # how many chunks from RAG to feed into the single LLM run
MAX_CHARS = 60000             # safety cap to avoid overlong prompts

# -------------------------
# Utils
# -------------------------
def sanitize(text: str, max_chars: int = MAX_CHARS) -> str:
    return text[:max_chars]

def parse_markdown_table(md: str):
    """Parse a markdown table into (header, rows). Returns header list and list[dict]."""
    lines = [l for l in md.strip().splitlines() if l.strip()]
    if not lines:
        return [], []
    # find header & separator
    header_idx = None
    for i, l in enumerate(lines):
        if "|" in l:
            if i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
                header_idx = i
                break
    if header_idx is None:
        return [], []
    header = [h.strip() for h in lines[header_idx].split("|") if h.strip()]
    data_lines = []
    for l in lines[header_idx + 2 :]:
        if "|" not in l:
            continue
        parts = [p.strip() for p in l.split("|")]
        # remove empty first/last caused by leading/trailing |
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) != len(header):
            continue
        data_lines.append(dict(zip(header, parts)))
    return header, data_lines

# -------------------------
# App
# -------------------------
st.title("One-shot RAG ➜ Derived Requirements + Ambiguity + SPI (characteristic)")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "results" not in st.session_state:
    st.session_state["results"] = []

uploaded_file = st.file_uploader("Upload traffic law (.txt)", type=["txt"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    st.success("Document uploaded.")

    with st.spinner("Indexing with FAISS (RAG)…"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=raw_text)]
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)

    query = st.text_input(
        "Query (what parts of the law do you want to derive requirements & SPIs from?)",
        value="All rules that constrain the behavior of an autonomous vehicle, including stopping, yielding, signaling, parking, and interaction with pedestrians."
    )

    if st.button("Run single-shot derivation"):
        with st.spinner("Retrieving relevant law text…"):
            retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
            retrieved_docs = retriever.get_relevant_documents(query)
            joined = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)
            law_context = sanitize(joined)

        with st.spinner("LLM deriving requirements, ambiguity & SPI (single run)…"):
            prompt = f"""
You are a traffic safety and functional safety expert for autonomous driving systems.

**Goal**:
From the legal text (California Vehicle Code or similar), extract only requirements that apply to the **autonomous vehicle/system behavior**.
Ignore obligations on third parties (e.g., pedestrians must wear helmets, cyclists must do X, administrative rules, penalty clauses).

For each relevant section:
1) Provide the **CVC Section** (e.g., §22500(a)) if present or infer it from the text.
2) Provide a concise **Derived Requirement** phrased as an implementation-neutral, deterministic requirement (using “shall”).
3) Provide **Ambiguity of requirement**: short note on where the law is vague or needs technical interpretation (timing, distances, thresholds, sensor limitations, etc.).
4) Provide an **SPI** (Safety Performance Indicator) as a **measurable characteristic** (NOT another requirement). Example: “Number of red-light violations per 1000 km”, “Mean time to yield before crosswalk entry (s)”, “Percentage of correct lane selection before intersection (%)”.
5) Provide **Advice / Reference**: point to related sections (e.g., emergency lights section) or standards (FMVSS, MUTCD, UNECE, ISO 26262/21448, UL 4600) that should be consulted to clarify ambiguity or complete implementation.

**Output ONLY a markdown table** with **exactly these columns** in this order:

| CVC Section | Derived Requirement | Ambiguity of requirement | SPI | Advice / Reference |

Text to analyze:
\"\"\" 
{law_context}
\"\"\" 
"""
            completion = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an expert in traffic law, safety engineering, and SPI definition for autonomous vehicles."},
                    {"role": "user", "content": prompt}
                ]
            )
            md_table = completion["choices"][0]["message"]["content"]

        st.markdown("### LLM Output (raw markdown)")
        st.markdown(md_table)

        header, rows = parse_markdown_table(md_table)
        if not rows:
            st.error("Could not parse a valid table from model output. You may need to refine the prompt or increase k.")
        else:
            # Ensure the expected columns exist
            expected = ["CVC Section", "Derived Requirement", "Ambiguity of requirement", "SPI", "Advice / Reference"]
            for col in expected:
                if col not in rows[0]:
                    st.error(f"Column '{col}' missing in LLM output. Please re-run or tweak the prompt.")
                    st.stop()

            st.session_state["results"] = rows
            st.success(f"Extracted {len(rows)} rows.")

# Show & export
if st.session_state["results"]:
    st.markdown("### Final Table")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📤 Export CSV",
        data=csv,
        file_name="cvc_requirements_with_spi_and_ambiguity.csv",
        mime="text/csv"
    )
