# rag_bot.py
import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ---------- NEW, CORRECT IMPORTS ----------
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# -------------------------------------------

# ------------------- CONFIG -------------------
DOCS_FOLDER   = "docs"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 200
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama3:8b"
TOP_K         = 4
DB_PATH       = "./chroma_db"

# ------------------- 1. LOAD & SPLIT -------------------
def load_and_split():
    docs = []
    for path in glob.glob(f"{DOCS_FOLDER}/*"):
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        print("No documents found!")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# ------------------- 2. BUILD VECTOR STORE -------------------
def build_vectorstore():
    chunks = load_and_split()
    if not chunks:
        return None

    print(f"Split into {len(chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector store built.")
    return vectorstore

# ------------------- 3. CREATE RAG CHAIN (MODERN) -------------------
def create_rag_chain():
    embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore  = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever    = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the following context.
If the answer is not in the context, say: "I cannot find this information in the document."

<context>
{context}
</context>

Question: {input}

Answer (concise, use bullet points if helpful):"""
    )

    # 1. Stuff documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 2. Full retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# ------------------- 4. QUERY -------------------
def ask(rag_chain, query):
    result = rag_chain.invoke({"input": query})
    answer = result["answer"]
    sources = list(set([
        doc.metadata.get("source", "").split("/")[-1]
        for doc in result.get("context", [])
    ]))
    return answer, sources

# ------------------- MAIN -------------------
if __name__ == "__main__":
    print("Building vector store (first run only)...")
    build_vectorstore()

    print("Initializing RAG chain...")
    rag_chain = create_rag_chain()

    print("\nRAG Bot Ready! Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        answer, sources = ask(rag_chain, q)
        print(f"\n{answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print("-" * 60)