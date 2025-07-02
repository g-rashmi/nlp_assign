from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import shutil


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pdf_chunks = {}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    pdf_chunks[file.filename] = chunks

    return {"message": "PDF uploaded and processed", "filename": file.filename}


@app.post("/ask")
async def ask_question(filename: str = Form(...), question: str = Form(...)):
    if filename not in pdf_chunks:
        return {"error": "Text not found. Please upload the PDF again."}

    docs = pdf_chunks[filename]

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using the context below.

Context:
{context}

Question: {question}
Answer:"""
    )

    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    result = chain.invoke({
        "input_documents": docs,
        "question": question
    })

    return {"answer": result['output_text']} ;