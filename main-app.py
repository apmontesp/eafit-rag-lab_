import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import PyPDF2

# Configuración de la página
st.set_page_config(layout="wide", page_title="RAG vs LLM Lab")
st.title("🧪 Laboratorio: RAG vs. LLM (Zero-shot)")

# Sidebar: Hiperparámetros (Fase 2)
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Model Select", ["llama3-70b-8192", "mixtral-8x7b-32768"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.5)
    chunk_size = st.slider("Chunk Size", 20, 2000, 500)
    top_k = st.slider("Top-K", 1, 10, 3)

# Lógica de Ingesta (Fase 2.1)
uploaded_file = st.file_uploader("Sube un PDF", type="pdf")

if uploaded_file and api_key:
    # Leer PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Chunking & Embeddings (Fase 3.1)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Área de pregunta
    user_question = st.text_input("Haz una pregunta sobre el documento:")

    if user_question:
        llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temp)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("LLM Simple")
            # Inferencia sin contexto
            response = llm.invoke(user_question)
            st.write(response.content)

        with col2:
            st.subheader("RAG Estándar")
            # Simulación de parámetros default (ej. chunk 500, k=3)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            st.write(qa_chain.run(user_question))

        with col3:
            st.subheader("RAG Optimizado")
            # RAG con el ajuste del Sidebar
            custom_prompt = PromptTemplate(
                template="Responde solo usando el contexto. Si no sabes, di 'No sé'. \nContexto: {context}\nPregunta: {question}",
                input_variables=["context", "question"]
            )
            qa_chain_opt = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, 
                chain_type_kwargs={"prompt": custom_prompt}
            )
            st.write(qa_chain_opt.run(user_question))
