from fastapi import FastAPI, UploadFile, File, Form
import fitz  # PyMuPDF
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = FastAPI()

# Load your custom QA model
tokenizer = AutoTokenizer.from_pretrained("2KKLabs/Kaleidoscope_large_v1")
model = AutoModelForQuestionAnswering.from_pretrained("2KKLabs/Kaleidoscope_large_v1")

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def ask_question(question: str, context: str) -> str:
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer_ids = inputs["input_ids"][0][start:end]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)
    with open("temp/last_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return {"message": "PDF uploaded and text extracted."}

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

@app.post("/ask")
async def ask_from_pdf(question: str = Form(...)):
    try:
        with open("temp/last_text.txt", "r", encoding="utf-8") as f:
            context = f.read()
    except FileNotFoundError:
        return {"error": "No PDF uploaded yet."}

    context = context[:3000]  # basic truncation
    answer = ask_question(question, context)

    return {"question": question, "answer": answer}
