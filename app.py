import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import GeminiRAG, DEFAULT_GENERATION_MODEL, DEFAULT_EMBEDDING_MODEL

# Load .env file if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatInput {
        bottom: 20px;
    }
    .chunk-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #1E88E5;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 12px;
        border-radius: 4px;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "indexed_chunk_count" not in st.session_state:
        st.session_state.indexed_chunk_count = 0
    if "last_indexed_time" not in st.session_state:
        st.session_state.last_indexed_time = None


init_session_state()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ การตั้งค่าระบบ")
    st.markdown("---")

    # 1. API Key handling
    env_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not env_api_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        env_api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    
    if env_api_key:
        st.success("🟢 อ่าน GEMINI_API_KEY จาก Environment แล้ว")
        api_key = env_api_key
    else:
        st.warning("⚠️ ไม่พบ GEMINI_API_KEY ใน Environment")
        api_key = st.text_input(
            "ระบุ Gemini API Key ด้วยตนเอง:",
            type="password",
            placeholder="AIzaSy...",
            help="สามารถสร้าง API Key ฟรีได้ที่ https://aistudio.google.com/app/apikey"
        ).strip()

    st.markdown("---")
    st.subheader("🧠 โมเดล & พารามิเตอร์")

    selected_model = st.selectbox(
        "LLM Model:",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-1.5-flash ตอบเร็ว เหมาะสำหรับ RAG ทั่วไป"
    )

    top_k = st.slider(
        "จำนวน Chunk ที่ดึงมา (Top-K):",
        min_value=1,
        max_value=10,
        value=3,
        help="จำนวนชิ้นส่วนเอกสารที่ใกล้เคียงที่สุดที่จะส่งให้ LLM ตอบ"
    )

    with st.expander("🛠️ ตั้งค่า Chunking ขั้นสูง"):
        chunk_size = st.number_input("Chunk Size (ตัวอักษร):", min_value=200, max_value=2000, value=600, step=100)
        chunk_overlap = st.number_input("Chunk Overlap (ตัวอักษร):", min_value=0, max_value=500, value=100, step=50)

    st.markdown("---")
    st.subheader("📁 คลังเอกสาร (docs/)")

    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(docs_dir) if not f.startswith(".")]

    if existing_files:
        st.write(f"พบ **{len(existing_files)}** ไฟล์ใน `{docs_dir}/`:")
        for fname in existing_files:
            fpath = os.path.join(docs_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            st.caption(f"• `{fname}` ({size_kb:.1f} KB)")
    else:
        st.info(f"ยังไม่มีเอกสารในโฟลเดอร์ `{docs_dir}/`")

    # Upload new file directly from UI
    uploaded_file = st.file_uploader("📤 เพิ่มเอกสารเข้า docs/", type=["txt", "md"])
    if uploaded_file is not None:
        save_path = os.path.join(docs_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"บันทึก `{uploaded_file.name}` เรียบร้อย!")
        st.rerun()

    # Button to re-index documents
    if st.button("🔄 ทำ Index เอกสารใหม่ (Re-index)", use_container_width=True, type="primary"):
        if not api_key:
            st.error("กรุณาระบุ Gemini API Key ก่อนทำ Index")
        else:
            with st.spinner("กำลังอ่านเอกสารและแปลงเป็น Embeddings..."):
                try:
                    rag = GeminiRAG(
                        api_key=api_key,
                        generation_model=selected_model,
                        embedding_model=DEFAULT_EMBEDDING_MODEL
                    )
                    progress_bar = st.progress(0.0)
                    chunk_count = rag.index_directory(
                        docs_dir=docs_dir,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        progress_callback=lambda p: progress_bar.progress(p)
                    )
                    progress_bar.empty()
                    st.session_state.rag_engine = rag
                    st.session_state.indexed_chunk_count = chunk_count
                    st.success(f"ทำ Index สำเร็จ! ทั้งหมด {chunk_count} chunks ในหน่วยความจำ")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการทำ Index: {str(e)}")

    if st.session_state.indexed_chunk_count > 0:
        st.caption(f"💾 หน่วยความจำ: **{st.session_state.indexed_chunk_count} chunks** พร้อมค้นหา")

    st.markdown("---")
    if st.button("🗑️ ล้างประวัติแชท (Clear Chat)", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------- MAIN CONTENT -----------------
st.title("🤖 Gemini RAG Assistant")
st.caption("ระบบถาม-ตอบอัจฉริยะจากเอกสารใน `docs/` ด้วย In-Memory Vector Search และ Gemini API")

# Ensure RAG engine instance is synced with current API key & model
if api_key:
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = GeminiRAG(
            api_key=api_key,
            generation_model=selected_model,
            embedding_model=DEFAULT_EMBEDDING_MODEL
        )
    else:
        st.session_state.rag_engine.set_api_key(api_key)
        st.session_state.rag_engine.generation_model_name = selected_model

# Auto-index on first launch if docs exist and not yet indexed
if (
    api_key
    and st.session_state.rag_engine
    and st.session_state.indexed_chunk_count == 0
    and existing_files
):
    with st.spinner("⚡ กำลังเตรียมข้อมูลและทำ Index เอกสารเริ่มต้นเข้าสู่หน่วยความจำ..."):
        try:
            chunks = st.session_state.rag_engine.index_directory(
                docs_dir=docs_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.session_state.indexed_chunk_count = chunks
            st.toast(f"พร้อมใช้งาน! โหลด {chunks} chunks เข้า Memory แล้ว", icon="🚀")
        except Exception as e:
            st.warning(f"ไม่สามารถทำ auto-index ได้: {e}")

# If no API key, prompt user
if not api_key:
    st.info("👈 กรุณาระบุ **GEMINI_API_KEY** ในแถบด้านซ้าย หรือตั้งค่าตัวแปรใน Environment / `.env` เพื่อเริ่มต้นใช้งาน")

# Display example questions if chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    ### 💡 ตัวอย่างคำถามที่สามารถลองถามได้:
    """)
    col1, col2, col3 = st.columns(3)
    sample_q1 = "Cloud Computing มี Service Models อะไรบ้าง?"
    sample_q2 = "EC2 กับ Launch Template เกี่ยวข้องกันอย่างไร?"
    sample_q3 = "ขั้นตอนการทำงานของ RAG มีอะไรบ้าง?"

    if col1.button(sample_q1, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": sample_q1})
        st.rerun()
    if col2.button(sample_q2, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": sample_q2})
        st.rerun()
    if col3.button(sample_q3, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": sample_q3})
        st.rerun()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are retrieved references attached to this message, display them
        if "references" in msg and msg["references"]:
            with st.expander(f"🔍 เอกสารอ้างอิง ({len(msg['references'])} chunks ที่ค้นพบ)"):
                for idx, ref in enumerate(msg["references"], 1):
                    score_pct = ref.get("score", 0) * 100
                    st.markdown(
                        f"**{idx}. [{ref.get('source')}]** - ความเกี่ยวข้อง: `{score_pct:.1f}%`"
                    )
                    st.info(ref.get("text", ""))

# Handle user input
user_query = st.chat_input("พิมพ์คำถามเกี่ยวกับเอกสารที่นี่...")

# Check if last message was from sample question click
pending_generation = False
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if len(st.session_state.messages) % 2 == 1:
        # User message waiting for response
        user_query_to_process = st.session_state.messages[-1]["content"]
        pending_generation = True

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    user_query_to_process = user_query
    pending_generation = True

if pending_generation:
    if not api_key:
        with st.chat_message("assistant"):
            st.error("กรุณาระบุ GEMINI_API_KEY ในแถบด้านข้างก่อนส่งคำถาม")
    elif (
        st.session_state.rag_engine is None
        or st.session_state.indexed_chunk_count == 0
    ):
        with st.chat_message("assistant"):
            st.warning("⚠️ ยังไม่มีข้อมูลใน Memory! กรุณากดปุ่ม **'🔄 ทำ Index เอกสารใหม่'** ในแถบด้านซ้าย")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔎 กำลังค้นหาข้อมูลในเอกสารและสร้างคำตอบ..."):
                try:
                    rag = st.session_state.rag_engine
                    
                    # Search and stream answer
                    response_stream, retrieved_chunks = rag.answer_query_stream(
                        user_query_to_process, top_k=top_k
                    )

                    # Prepare reference data for expander and storage
                    refs_data = []
                    for chunk, score in retrieved_chunks:
                        refs_data.append({
                            "source": chunk.source,
                            "text": chunk.text,
                            "score": float(score)
                        })

                    # Stream generation
                    def stream_generator():
                        for chunk in response_stream:
                            if hasattr(chunk, "text") and chunk.text:
                                yield chunk.text

                    full_response = st.write_stream(stream_generator())

                    # Show retrieved chunks expander
                    if refs_data:
                        with st.expander(f"🔍 เอกสารอ้างอิง ({len(refs_data)} chunks ที่ค้นพบ)"):
                            for idx, ref in enumerate(refs_data, 1):
                                score_pct = ref.get("score", 0) * 100
                                st.markdown(
                                    f"**{idx}. [{ref.get('source')}]** - ความเกี่ยวข้อง: `{score_pct:.1f}%`"
                                )
                                st.info(ref.get("text", ""))

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "references": refs_data
                    })

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการตอบคำถาม: {str(e)}")
