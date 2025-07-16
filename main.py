from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Initialize FastAPI
app = FastAPI()

# CORS config (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embedding and QA models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("2KKLabs/Kaleidoscope_large_v1")
model = AutoModelForQuestionAnswering.from_pretrained("2KKLabs/Kaleidoscope_large_v1")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Global stores
vectorstore = None
uploaded_files_info = []  # List of dicts: {filename, path}


# Helpers
def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)


def embed_documents(docs):
    return FAISS.from_documents(docs, embedding_model)


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vectorstore, uploaded_files_info
    all_chunks = []

    for file in files:
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = process_pdf(file_location)
        all_chunks.extend(chunks)

        uploaded_files_info.append({
            "filename": file.filename,
            "path": file_location
        })

    if vectorstore is None:
        vectorstore = embed_documents(all_chunks)
    else:
        vectorstore.add_documents(all_chunks)

    return {"message": f"{len(files)} PDF(s) uploaded and indexed successfully."}


@app.post("/ask")
async def ask_user_question(question: str = Form(...)):
    global vectorstore, uploaded_files_info

    if not uploaded_files_info:
        return {"error": "No PDFs uploaded yet."}

    q_lower = question.lower().strip()

    # Handle metadata-based questions
    if any(kw in q_lower for kw in ["title", "name of the pdf", "name of document", "file name"]):
        titles = [file["filename"] for file in uploaded_files_info]
        return {"question": question, "answer": ", ".join(titles)}

    if any(kw in q_lower for kw in ["list all", "uploaded files", "show all files"]):
        return {"question": question, "answer": [f["filename"] for f in uploaded_files_info]}

    if any(kw in q_lower for kw in ["department", "dept", "faculty", "school"]):
        doc_texts = []
        for f in uploaded_files_info:
            loader = PyPDFLoader(f["path"])
            pages = loader.load()
            doc_texts.append(" ".join([page.page_content for page in pages[:1]]))  # Just first page
        all_text = " ".join(doc_texts)
        result = qa_pipeline(question=question, context=all_text[:3000])
        return {
            "question": question,
            "answer": result["answer"],
            "score": result["score"]
        }

    if vectorstore is None:
        return {"error": "No vectorstore available."}

    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    result = qa_pipeline(question=question, context=context)

    return {
        "question": question,
        "answer": result["answer"],
        "score": result["score"],
        "context": context
    }


@app.get("/files")
def list_uploaded_files():
    return {"uploaded_files": [f["filename"] for f in uploaded_files_info]}


@app.get("/")
def root():
    return {"message": "FastAPI with LangChain + FAISS + Transformers is running."}
