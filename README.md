# 🤖 Gemini RAG Assistant (Python + Streamlit)

เว็บแอปพลิเคชัน RAG (Retrieval-Augmented Generation) พัฒนาด้วย **Python** และ **Streamlit** ขับเคลื่อนด้วย **Google Gemini API** (`gemini-3.8-flash` และ `gemini-embedding-001`) โดยอ่านเอกสารจากโฟลเดอร์ `docs/` ทำ Text Chunking, สร้าง Embeddings และเก็บข้อมูลเวกเตอร์ไว้ใน **In-Memory** พร้อมค้นหา Chunk ที่เกี่ยวข้องที่สุด (Cosine Similarity) แล้วส่งต่อให้ LLM ตอบคำถามได้อย่างแม่นยำ พร้อมแสดงเอกสารอ้างอิงและคะแนนความเกี่ยวข้อง (Similarity Score)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
RAG/
├── app.py                  # สคริปต์หลักสำหรับ Streamlit Web UI
├── rag_engine.py           # ระบบ RAG: อ่านไฟล์, Chunking, Embedding, In-Memory Search
├── docs/                   # โฟลเดอร์เก็บเอกสารความรู้ (.txt, .md, .pdf)
│   └── info.txt            # ตัวอย่างเอกสารความรู้ (Cloud Computing & RAG)
├── requirements.txt        # รายการ Python dependencies
├── render.yaml             # Render Blueprint สำหรับ Deploy บน Render อัตโนมัติ
├── Procfile                # คำสั่งรันสำหรับ Render Web Service
├── .env.example            # ไฟล์ตัวอย่างสำหรับการตั้งค่า Environment Variable
├── .gitignore              # ไฟล์ยกเว้น Git
├── .streamlit/
│   └── config.toml         # การตั้งค่า Streamlit สำหรับโหมด Headless & Render
└── README.md               # คู่มือการติดตั้งและ Deploy
```

---

## 🚀 จุดเด่นของระบบ (Features)

1. **In-Memory Vector Store**: ไม่ต้องติดตั้งฐานข้อมูลภายนอก คำนวณ Cosine Similarity รวดเร็วด้วย NumPy และมี Pure-Python Fallback
2. **Gemini Embeddings (`gemini-embedding-001`)**: ใช้ Task Type แยกกันตามมาตรฐานของ Google (`retrieval_document` สำหรับเอกสาร และ `retrieval_query` สำหรับคำถาม) เพิ่มความแม่นยำในการค้นหา
3. **Smart Chunking**: แบ่งเอกสารตามย่อหน้าและบรรทัด โดยมี Overlap เพื่อไม่ให้บริบทของข้อความขาดตอน
4. **Zero-Hallucination Prompting**: กำหนด Prompt สั่งให้ LLM ตอบจากเอกสารที่กำหนดเท่านั้น หากไม่มีข้อมูลให้แจ้งอย่างสุภาพ
5. **Streaming Response**: แสดงผลคำตอบแบบเรียลไทม์ (Streaming) คล้าย ChatGPT
6. **Reference Inspection**: สามารถคลิกขยายดูชิ้นส่วนเอกสารที่ดึงมา (Retrieved Chunks) พร้อมแสดงเปอร์เซ็นต์ความเกี่ยวข้อง
7. **Document Management**: รองรับการอัปโหลดไฟล์เอกสารใหม่ผ่านหน้าเว็บ และปุ่มกด Re-index
8. **Render-Ready**: รองรับตัวแปรพอร์ต `$PORT` และรันบน Render ได้ทันที

---

## 🛠️ วิธีการติดตั้งและรันในเครื่อง (Local Setup)

### 1. โคลนหรือเปิดโฟลเดอร์โปรเจกต์
```bash
cd RAG
```

### 2. สร้างและเปิดใช้งาน Virtual Environment (แนะนำ)
```bash
python3 -m venv .venv
source .venv/bin/activate    # สำหรับ macOS / Linux
# หรือ .venv\Scripts\activate สำหรับ Windows
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า Gemini API Key
คุณสามารถรับ API Key ได้ฟรีที่ [Google AI Studio](https://aistudio.google.com/app/apikey)

คัดลอกไฟล์ `.env.example` เป็น `.env`:
```bash
cp .env.example .env
```
แล้วเปิดไฟล์ `.env` เพื่อใส่ Key ของคุณ:
```env
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxx
```
*(หรือจะกรอกผ่านช่องใส่รหัสในแถบด้านซ้ายของเว็บแอปก็ได้เช่นกัน)*

### 5. วางเอกสารที่ต้องการ
นำไฟล์ข้อความ `.txt`, `.md` หรือ `.pdf` ไปวางในโฟลเดอร์ `docs/` (มีไฟล์ตัวอย่าง `docs/info.txt` ให้พร้อมใช้งาน)

### 6. รันเว็บแอป
```bash
streamlit run app.py
```
เปิดเบราว์เซอร์ไปที่ `http://localhost:8501` เพื่อเริ่มต้นใช้งาน

---

## 🌐 วิธีการ Deploy บน Render (Step-by-Step)

เว็บแอปนี้พร้อมสำหรับการ Deploy บน [Render](https://render.com) ทันที มีขั้นตอนง่ายๆ ดังนี้:

### วิธีที่ 1: Deploy ผ่าน Render Web Service (แนะนำ)

1. **Push โค้ดขึ้น GitHub / GitLab** ของคุณ
2. เข้าสู่ระบบ [Render Dashboard](https://dashboard.render.com)
3. กด **New +** > เลือก **Web Service**
4. เลือก Repository ของคุณ แล้วตั้งค่าดังนี้:
   - **Name**: `gemini-rag-app` (หรือชื่อที่ต้องการ)
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: 
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
5. ในส่วน **Environment Variables** ให้กด **Add Environment Variable**:
   - `GEMINI_API_KEY`: ใส่ API Key ของคุณที่ได้จาก Google AI Studio
   - `PYTHON_VERSION`: `3.11.9`
6. กด **Create Web Service**
7. รอ Render ทำการ Build และ Deploy สักครู่ เมื่อเสร็จแล้วจะได้รับ URL เข้าใช้งานทันที 🎉

---

### วิธีที่ 2: Deploy ผ่าน Render Blueprint (IaC)

โปรเจกต์นี้มีไฟล์ `render.yaml` อยู่แล้ว คุณสามารถใช้ฟีเจอร์ Blueprint ได้โดยตรง:
1. Push โค้ดขึ้น GitHub
2. บน Render Dashboard กด **New +** > เลือก **Blueprint**
3. เลือก Repository ระบบจะอ่านการตั้งค่าจาก `render.yaml` อัตโนมัติ
4. กรอกค่า `GEMINI_API_KEY` ในช่องที่ระบบร้องขอ แล้วกดยืนยันการ Deploy

---

## ⚙️ พารามิเตอร์ที่สามารถปรับแต่งได้ในหน้าเว็บ

- **LLM Model**: ค่าเริ่มต้น `gemini-3.8-flash` หรือเลือกโมเดลอื่นที่ API Key ของคุณใช้งานได้ (ดึงรายการอัตโนมัติ)
- **Top-K Chunks**: จำนวนส่วนของเอกสารที่ใกล้เคียงที่สุดที่จะส่งให้ LLM (ค่าเริ่มต้นคือ 3)
- **Chunk Size / Overlap**: ขนาดความยาวตัวอักษรของแต่ละ chunk และระยะซ้อนทับ
- **Re-index Button**: กดเพื่อสร้าง Index ใหม่เมื่อมีการแก้ไขหรือเพิ่มเอกสารใน `docs/`
