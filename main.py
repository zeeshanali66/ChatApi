import os
from flask import Flask, request, jsonify
from pydantic import BaseModel  # Import BaseModel from pydantic
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# === CONFIG ===
PDF_FOLDER = 'D:/University/FYP/Code/ChatApi/pdfs'  # Path to your PDFs folder
DB_PATH = 'D:/University/FYP/Code/ChatApi/chroma_db'  # Path to store vector DB
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_MODEL = 'llama3-70b-8192'  # Change to your model, "llama-3.3-70b-versatile" is not official

# === Flask app ===
app = Flask(__name__)

# === 1. Initialize LLM ===
def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY") or "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"
    if groq_api_key == "your-groq-api-key":
        raise ValueError("⚠️ Please set your Groq API key in the code or as an environment variable.")
    
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL
    )
    return llm

# === 2. Create Vector DB ===
def create_vector_db():
    print("📄 Loading and chunking PDFs...")
    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"The specified folder '{PDF_FOLDER}' does not exist. Please provide a valid path.")

    # Check if there are any PDF files in the folder
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the folder '{PDF_FOLDER}'. Please ensure that the folder contains PDFs.")

    # Proceed to load the documents
    loader = DirectoryLoader(PDF_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    print("🔍 Creating embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    print("💾 Saving to Chroma vector DB...")
    vector_db = Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    vector_db.persist()

    return vector_db

# === 3. Setup QA Chain ===
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()

    prompt_template = """
You are a compassionate mental health chatbot. Use the context to answer the user's question kindly.

Context:
{context}

User: {question}
Chatbot:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# === 4. Startup: Load LLM and Vector DB once ===
llm = initialize_llm()
if not os.path.exists(DB_PATH):
    print("📂 Creating a new Chroma vector DB...")
    vector_db = create_vector_db()
else:
    print("📂 Loading existing Chroma vector DB...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# === 5. Flask endpoint ===
class ChatRequest(BaseModel):  # Defining the request body with BaseModel
    question: str

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        user_input = data['prompt']  # Get the prompt from the user

        # Run the question through the QA chain and get the response
        response = qa_chain.run(user_input)

        # Return the response as JSON
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def root():
    return jsonify({"message": "Mental Health Chatbot API is running."})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
