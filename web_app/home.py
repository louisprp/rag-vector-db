import streamlit as st
import textwrap
import os

from pinecone import Pinecone

from sentence_transformers import SentenceTransformer
from pinecone.core.client.model.query_response import QueryResponse

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

COMPLETIONS_MODEL = "llama3-8b-8192"
EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 500,
    "model": COMPLETIONS_MODEL,
    "stream": True
}

st.set_page_config("Home", initial_sidebar_state="collapsed")

retriever = SentenceTransformer(EMBEDDINGS_MODEL)

# Connect to pinecone environment
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "abstractive-question-answering"
# Connect to abstractive-question-answering index we created
index = pc.Index(index_name)


def query_pinecone(query, top_k, sources):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    if not sources:
        xc = index.query(vector=xq, top_k=top_k, include_metadata=True)
    else:
        xc = index.query(vector=xq, top_k=top_k, include_metadata=True, filter={"source": { "$in" : sources }})
    return xc

client = Groq()

# Construct the OpenAI API prompt
def construct_prompt(question: str, results):
    header = """You are an AI assistant providing helpful advice. Extensively answer the provided question as truthfully and detailed as possible using the provided context and sources, and if the answer is not contained within the text below, say "I'm sorry but I can't provide an answer."\n\nContext:\n"""

    full_context = ""
    for res in results["matches"]:
        full_context += res.metadata["page_content"]
        full_context += " Source: " + res.metadata["source"] + "\n\n"
    
    prompt = header + "".join(full_context) + "\n\n Q: " + question + "\n A:"

    return client.chat.completions.create(messages=[
        {
            "role": "user",
            "content": prompt
        }
    ], **COMPLETIONS_API_PARAMS)


# Main content
st.title("Search")

query = st.text_input("Query")
options = st.multiselect(
    "Filter by source",
    [
        # Put your sources here
    ],
)
generate_answer = st.checkbox("Generate Answer?", value=False)

if query:
    results = query_pinecone(query, 5, options)

    if generate_answer:
        st.subheader("Answer")

        res_box = st.empty()
        report = []
        # Looping over the response
        for resp in construct_prompt(query, results):
            report.append(resp.choices[0].delta.content or "")
            result = "".join(report).strip()
            # result = result.replace("\n", "")        
            res_box.markdown(f'{result}') 

    st.subheader("Sources")

    for res in results["matches"]:
        st.caption(res.metadata["source"])
        content = res.metadata["page_content"]
        st.markdown(textwrap.fill(content, 100))
        with st.expander("Show more", expanded=False):
            next_id = str(int(res.id) + 1)
            next_content = index.fetch([next_id])["vectors"][next_id]["metadata"][
                "page_content"
            ]
            st.markdown(textwrap.fill(next_content, 100))
