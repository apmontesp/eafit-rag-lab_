import streamlit as st
import time
import base64
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import PyPDF2

# Configuración de página y estilo
st.set_page_config(layout="wide", page_title="EAFIT - RAG vs LLM Lab")
st.title("🧪 RAG vs. LLM: Laboratorio Experimental")

# --- SIDEBAR: Configuración (Fase 2) ---
with st.sidebar:
    st.header("⚙️ Hiperparámetros")
    api_key = st.text_input("Groq API Key", type="password")
    model_choice = st.selectbox("Model Select", ["llama3-70b-8192", "mixtral-8x7b-32768"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.1)
    c_size = st.slider("Chunk Size", 20, 2000, 500)
    top_k = st.slider("Top-K", 1, 10, 3)
    
    st.divider()
    st.info("Configura estos valores para observar el cambio en la columna 'RAG Optimizado'.")

# --- FUNCIONES DE SOPORTE ---
def get_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join([page.extract_text() for page in reader.pages])

def get_image_text(image_file, api_key):
    # Fase 2.1: OCR usando Llama 3.2 Vision
    client = ChatGroq(groq_api_key=api_key, model_name="llama-3.2-11b-vision-preview")
    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    msg = client.invoke([
        {"role": "user", "content": [
            {"type": "text", "text": "Extract all text from this image accurately."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]}
    ])
    return msg.content

# --- FASE 2: INGESTA ---
uploaded_file = st.file_uploader("Sube un PDF o Imagen para analizar", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file and api_key:
    # 1. Extracción de texto
    if uploaded_file.type == "application/pdf":
        raw_text = get_pdf_text(uploaded_file)
    else:
        raw_text = get_image_text(uploaded_file, api_key)

    # 2. Chunking & Embeddings (Fase 3.1)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # --- FASE 3 & 4: COMPARACIÓN Y MÉTRICAS ---
    query = st.text_input("Introduce tu pregunta sobre el documento:")
    
    if query:
        llm = ChatGroq(groq_api_key=api_key, model_name=model_choice, temperature=temp)
        
        col1, col2, col3 = st.columns(3)

        # Columna 1: LLM Simple
        with col1:
            st.subheader("🤖 LLM Simple")
            start = time.time()
            resp1 = llm.invoke(query).content
            latency = time.time() - start
            st.write(resp1)
            st.caption(f"⏱️ Latencia: {latency:.2f}s")

        # Columna 2: RAG Estándar (Parámetros por defecto)
        with col2:
            st.subheader("📚 RAG Estándar")
            start = time.time()
            # Usamos k=3 por defecto
            std_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_std = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=std_retriever)
            resp2 = qa_std.run(query)
            latency = time.time() - start
            st.write(resp2)
            st.caption(f"⏱️ Latencia: {latency:.2f}s")

        # Columna 3: RAG Optimizado (Ajuste del Sidebar)
        with col3:
            st.subheader("🚀 RAG Optimizado")
            start = time.time()
            opt_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            
            # System Prompt para evitar alucinaciones (Fase 4.2)
            opt_prompt = PromptTemplate(
                template="Contexto: {context}\n\nPregunta: {question}\n\nResponde estrictamente basado en el contexto. Si la respuesta no está, di 'No sé'.",
                input_variables=["context", "question"]
            )
            
            qa_opt = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=opt_retriever,
                chain_type_kwargs={"prompt": opt_prompt},
                return_source_documents=True
            )
            
            result = qa_opt({"query": query})
            latency
