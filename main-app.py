import streamlit as st
import time
import numpy as np
import io
import base64
import re

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Playground · EAFIT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #2a2a40;
}
[data-testid="stSidebar"] * {
    color: #c9c7d8 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

/* Column cards */
.result-card {
    background: #13131e;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    min-height: 260px;
    color: #dcdaf0;
    white-space: pre-wrap;
    word-break: break-word;
}

.metric-badge {
    display: inline-block;
    background: #1e1e30;
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #7b78c8;
    margin-top: 8px;
}

.col-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.col-header-1 { color: #ff6b6b; }
.col-header-2 { color: #4ecdc4; }
.col-header-3 { color: #a78bfa; }

.pill {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.pill-1 { background: #2a1010; color: #ff6b6b; border: 1px solid #ff6b6b44; }
.pill-2 { background: #0a1f1e; color: #4ecdc4; border: 1px solid #4ecdc444; }
.pill-3 { background: #1a1030; color: #a78bfa; border: 1px solid #a78bfa44; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: #13131e !important;
    border: 1px solid #2a2a40 !important;
    color: #e8e6f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0.55rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Divider */
hr { border-color: #2a2a40 !important; }

/* Expander */
details {
    background: #13131e;
    border: 1px solid #2a2a40 !important;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_groq_client():
    """Return a Groq client using the stored API key."""
    try:
        from groq import Groq
        key = st.session_state.get("groq_api_key", "")
        if not key:
            return None
        return Groq(api_key=key)
    except ImportError:
        st.error("groq package not found. Add `groq` to requirements.txt")
        return None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2."""
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def extract_text_from_image_groq(file_bytes: bytes, client) -> str:
    """Use Groq vision model to OCR an image."""
    b64 = base64.b64encode(file_bytes).decode()
    # detect mime type naively
    mime = "image/jpeg"
    if file_bytes[:4] == b"\x89PNG":
        mime = "image/png"

    resp = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract ALL text you can see in this image. Return only the extracted text, nothing else.",
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


def chunk_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    """Simple recursive character text splitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(overlap, chunk_size // 4),
        length_function=len,
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(chunks: list[str]):
    """Build a FAISS index from text chunks."""
    import faiss
    embedder = load_embedder()
    vecs = embedder.encode(chunks, show_progress_bar=False).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalize)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, vecs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def retrieve(query: str, index, chunks: list[str], top_k: int):
    """Return (top_k chunks, similarity scores)."""
    import faiss
    embedder = load_embedder()
    q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, top_k)
    results = [(chunks[i], float(scores[0][j])) for j, i in enumerate(ids[0]) if i < len(chunks)]
    return results


def llm_simple(query: str, model: str, temperature: float, client) -> tuple[str, float]:
    """Zero-shot LLM call, no context."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=1024,
        )
        elapsed = time.time() - t0
        return resp.choices[0].message.content.strip(), elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return f"❌ Error al llamar la API: {str(e)}", elapsed


def llm_rag(query: str, context_chunks: list[tuple], model: str, temperature: float,
            client, strict: bool = False) -> tuple[str, float, float]:
    """RAG-augmented LLM call. Returns (answer, elapsed, avg_similarity)."""
    context = "\n\n---\n\n".join([c for c, _ in context_chunks])
    avg_sim = float(np.mean([s for _, s in context_chunks])) if context_chunks else 0.0

    system_msg = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, reply exactly: 'No sé. La información no está en el documento.'"
        if strict else
        "You are a helpful assistant. Use the provided context to answer accurately."
    )

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        elapsed = time.time() - t0
        return resp.choices[0].message.content.strip(), elapsed, avg_sim
    except Exception as e:
        elapsed = time.time() - t0
        return f"❌ Error al llamar la API: {str(e)}", elapsed, avg_sim


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 RAG Lab · EAFIT")
    st.markdown("---")

    groq_api_key = st.text_input(
        "🔑 Groq API Key", type="password",
        placeholder="gsk_...",
        help="Obtén tu clave gratis en console.groq.com"
    )
    if groq_api_key:
        st.session_state["groq_api_key"] = groq_api_key

    st.markdown("### ⚙️ Hiperparámetros")

    model_choice = st.selectbox(
        "🤖 Modelo",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        help="Modelos disponibles en Groq (2025)"
    )

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2, 0.05)
    chunk_size = st.slider("📦 Chunk Size (tokens aprox.)", 20, 2000, 500, 10)
    top_k = st.slider("🔍 Top-K fragmentos", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 📁 Documento")
    uploaded_file = st.file_uploader(
        "Sube un PDF o imagen",
        type=["pdf", "png", "jpg", "jpeg"],
        help="PDF: extraído con PyPDF2. Imagen: OCR con Llama Vision"
    )

    process_btn = st.button("⚡ Procesar documento", use_container_width=True)

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("# RAG Playground")
st.markdown(
    "<span style='font-family:Space Mono;font-size:0.8rem;color:#7b78c8'>"
    "Maestría en Ciencia de Datos · EAFIT · Taller 03 — LLM vs RAG</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Document Processing ──────────────────────────────────────────────────────
if process_btn:
    if not st.session_state.get("groq_api_key"):
        st.error("⚠️ Ingresa tu Groq API Key en el sidebar.")
    elif uploaded_file is None:
        st.warning("⚠️ Sube un archivo primero.")
    else:
        client = get_groq_client()
        file_bytes = uploaded_file.read()
        fname = uploaded_file.name.lower()

        with st.spinner("📖 Extrayendo texto..."):
            if fname.endswith(".pdf"):
                raw_text = extract_text_from_pdf(file_bytes)
                st.session_state["ocr_method"] = "PyPDF2"
            else:
                raw_text = extract_text_from_image_groq(file_bytes, client)
                st.session_state["ocr_method"] = "Llama-3.2-11b-vision (OCR)"

        if not raw_text:
            st.error("No se pudo extraer texto del archivo.")
        else:
            with st.spinner("✂️ Chunking & embeddings..."):
                chunks = chunk_text(raw_text, chunk_size)
                index, vecs = build_faiss_index(chunks)

            st.session_state["raw_text"] = raw_text
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = index
            st.session_state["faiss_vecs"] = vecs
            st.session_state["doc_ready"] = True
            st.session_state["doc_name"] = uploaded_file.name

            st.success(
                f"✅ Documento listo: **{len(chunks)} chunks** | "
                f"Método: {st.session_state['ocr_method']}"
            )

# ── Query Area ───────────────────────────────────────────────────────────────
st.markdown("### 💬 Pregunta")
query = st.text_input(
    "",
    placeholder="Escribe tu pregunta sobre el documento...",
    label_visibility="collapsed"
)
run_btn = st.button("🚀 Comparar respuestas", use_container_width=False)

if run_btn:
    if not query.strip():
        st.warning("Escribe una pregunta.")
    elif not st.session_state.get("groq_api_key"):
        st.error("Ingresa tu Groq API Key.")
    elif not st.session_state.get("doc_ready"):
        st.warning("Procesa un documento primero.")
    else:
        client = get_groq_client()
        chunks = st.session_state["chunks"]
        index = st.session_state["faiss_index"]

        col1, col2, col3 = st.columns(3)

        # ── Column 1: LLM Simple ─────────────────────────────────────────────
        with col1:
            st.markdown('<div class="col-header col-header-1">① LLM Simple</div>', unsafe_allow_html=True)
            st.markdown('<div class="pill pill-1">Zero-shot · Sin contexto</div>', unsafe_allow_html=True)
            with st.spinner("Generando..."):
                ans1, t1 = llm_simple(query, model_choice, temperature, client)
            st.markdown(f'<div class="result-card">{ans1}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-badge">⏱ {t1:.2f}s</div>'
                f'<div class="metric-badge">🔵 Similitud: N/A</div>',
                unsafe_allow_html=True,
            )

        # ── Column 2: RAG Estándar ────────────────────────────────────────────
        with col2:
            st.markdown('<div class="col-header col-header-2">② RAG Estándar</div>', unsafe_allow_html=True)
            st.markdown('<div class="pill pill-2">top_k=3 · chunk=500 · temp=0.2</div>', unsafe_allow_html=True)

            # use fixed defaults for "standard"
            with st.spinner("Recuperando & generando..."):
                default_results = retrieve(query, index, chunks, top_k=3)
                ans2, t2, sim2 = llm_rag(query, default_results, model_choice, 0.2, client, strict=False)

            st.markdown(f'<div class="result-card">{ans2}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-badge">⏱ {t2:.2f}s</div>'
                f'<div class="metric-badge">📐 Similitud coseno: {sim2:.4f}</div>',
                unsafe_allow_html=True,
            )

        # ── Column 3: RAG Optimizado ──────────────────────────────────────────
        with col3:
            st.markdown('<div class="col-header col-header-3">③ RAG Optimizado</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pill pill-3">top_k={top_k} · chunk={chunk_size} · temp={temperature}</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Recuperando & generando..."):
                opt_results = retrieve(query, index, chunks, top_k=top_k)
                ans3, t3, sim3 = llm_rag(query, opt_results, model_choice, temperature, client, strict=True)

            st.markdown(f'<div class="result-card">{ans3}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-badge">⏱ {t3:.2f}s</div>'
                f'<div class="metric-badge">📐 Similitud coseno: {sim3:.4f}</div>',
                unsafe_allow_html=True,
            )

        # ── Context Preview ───────────────────────────────────────────────────
        with st.expander("🔍 Ver fragmentos recuperados (RAG Optimizado)"):
            for i, (chunk, score) in enumerate(opt_results):
                st.markdown(
                    f"**Fragmento {i+1}** — similitud coseno: `{score:.4f}`\n\n"
                    f"```\n{chunk[:500]}{'...' if len(chunk) > 500 else ''}\n```"
                )

# ── Analysis Section ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 Fase 4 · Análisis de Conceptos")

with st.expander("① Alucinación: ¿Cuándo inventa el LLM Simple?"):
    st.markdown("""
**¿Qué es la alucinación?**
Un LLM sin contexto responde con su conocimiento paramétrico (memorizado durante el entrenamiento).
Cuando el documento contiene datos específicos —nombres, fechas, cifras, terminología técnica—
el modelo puede *inventar* respuestas plausibles pero incorrectas porque no accede al documento real.

**Ejemplo típico:**
> Usuario pregunta: *"¿Cuál es el umbral de temperatura del proceso X?"*
> LLM Simple: inventa `120 °C` (suena razonable, pero el doc dice `85 °C`).
> RAG: recupera el fragmento exacto → responde `85 °C`.

**Conclusión:** el LLM Simple alucina con mayor probabilidad en datos cuantitativos,
nombres propios y procedimientos específicos que no forman parte de su entrenamiento general.
    """)

with st.expander("② Inyección de Contexto: System Prompt estricto"):
    st.markdown("""
**RAG Optimizado ya incluye inyección de contexto estricto.**

El *system prompt* del RAG Optimizado dice literalmente:

```
"Si la respuesta no está en el contexto, responde exactamente:
'No sé. La información no está en el documento.'"
```

**Efecto observable:**
- Con contexto relevante → respuesta precisa del fragmento.
- Sin contexto relevante → el modelo admite su ignorancia en lugar de alucinar.

**¿Por qué funciona?** El system prompt es procesado antes del turno del usuario
y actúa como una restricción de distribución: el modelo aprende, durante el prompt,
que inventar información penaliza la coherencia con el sistema.
    """)

with st.expander("③ Fine-Tuning vs RAG: ¿Por qué RAG gana aquí?"):
    st.markdown("""
| Criterio | Fine-Tuning | RAG |
|---|---|---|
| Costo | Alto (GPU, tiempo, datos etiquetados) | Bajo (solo embeddings + inferencia) |
| Actualización | Requiere reentrenar | Sólo actualizar el vector store |
| Transparencia | Caja negra | Fragmentos recuperables y auditables |
| Privacidad | Datos en el modelo | Datos en tu infraestructura |
| Latencia | Igual al modelo base | Agrega tiempo de recuperación (~ms) |

**Para este taller:** el documento es privado, cambia con cada sesión y no existe un
corpus de entrenamiento. Fine-Tuning requeriría miles de pares pregunta-respuesta
del mismo dominio. RAG resuelve el problema en minutos sin ningún dato de entrenamiento adicional.
    """)

with st.expander("④ Transformer vs No-Transformer en Embeddings"):
    st.markdown("""
**¿Los embeddings de `all-MiniLM-L6-v2` dependen de Transformers?**

**Sí.** El modelo `all-MiniLM-L6-v2` es una versión destilada de BERT (Bidirectional Encoder
Representations from Transformers). Su arquitectura usa:

- **Self-Attention** multi-cabeza para capturar relaciones contextuales entre tokens.
- **Positional Encoding** para preservar el orden de las palabras.
- **Feed-Forward Layers** por cada bloque Transformer.

La capa final produce un vector denso de 384 dimensiones que codifica el *significado semántico*
del texto. Sin la arquitectura Transformer, no sería posible capturar dependencias de largo alcance
entre palabras (ej. pronombres que refieren a sustantivos lejanos).

**Alternativas no-Transformer:** TF-IDF, BM25, Word2Vec (word-level, sin contexto).
Estos generan embeddings más simples pero no comprenden el contexto de la oración completa.
    """)

with st.expander("📊 Reto: Similitud de Coseno — ¿Qué significa?"):
    st.markdown("""
**Fórmula:**

$$\\text{sim}(q, d) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\| \\cdot \\|\\mathbf{d}\\|}$$

- **1.0** → vectores idénticos (pregunta = fragmento).
- **0.0** → sin relación semántica.
- **< 0** → significados opuestos.

**En la práctica:** fragmentos con similitud > 0.4 suelen ser relevantes.
Los valores se muestran en `📐 Similitud coseno` bajo cada respuesta RAG.

La similitud se calcula **entre el embedding de la pregunta** y el embedding
de cada fragmento recuperado. El promedio de los Top-K fragmentos
es el valor mostrado en el dashboard.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-family:Space Mono;font-size:0.7rem;color:#3a3a5c'>"
    "EAFIT · Maestría Ciencia de Datos · Taller 03 · Prof. Jorge Iván Padilla Buriticá, Ph.D."
    "</div>",
    unsafe_allow_html=True,
)
