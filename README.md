# 🌸 Sebelas Decor — RAG Chatbot

> **Retrieval-Augmented Generation (RAG) Chatbot** untuk bisnis jasa dekorasi event **Sebelas Decor**.  
> Chatbot ini menjawab pertanyaan pelanggan secara otomatis berdasarkan knowledge base (pricelist, katalog tema, FAQ) menggunakan Dense Vector Retrieval + LLM.

---

## 📐 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                       USER / PELANGGAN                      │
│                    (Browser / WhatsApp)                      │
└────────────────────────┬────────────────────────────────────┘
                         │  Pertanyaan
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FRONTEND (index.html)                      │
│         WhatsApp-style Chat UI (Dark Theme)                  │
│         Fetch API → POST /api/chat                           │
└────────────────────────┬────────────────────────────────────┘
                         │  JSON {"message": "..."}
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FLASK SERVER (app.py)                        │
│               http://127.0.0.1:5000                          │
│              Endpoint: POST /api/chat                        │
│                     + CORS                                   │
└────────────────────────┬────────────────────────────────────┘
                         │  query_rag(message)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                RAG ENGINE (rag_engine.py)                     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  1. LOAD     │    │  2. CHUNK    │    │  3. EMBED    │   │
│  │  Dokumen .txt│───▶│  500 char    │───▶│  MiniLM-L6   │   │
│  │  knowledge/  │    │  overlap 80  │    │  (lokal/CPU) │   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘   │
│                                                  │           │
│                                                  ▼           │
│                                          ┌──────────────┐   │
│                                          │  4. STORE    │   │
│                                          │  ChromaDB    │   │
│                                          │  (lokal)     │   │
│                                          └──────┬───────┘   │
│                                                  │           │
│  ┌──────────────┐    ┌──────────────┐           │           │
│  │  6. GENERATE │    │  5. SEARCH   │◀──────────┘           │
│  │  9router API │◀───│  Similarity  │                       │
│  │  (LLM)      │    │  Top-K = 4   │                       │
│  └──────┬───────┘    └──────────────┘                       │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │  Jawaban natural & ramah
          ▼
┌─────────────────────────────────────────────────────────────┐
│              RESPONS KE USER / PELANGGAN                     │
│         "Halo Kak! Berikut paket dekorasi kami..."          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Proyek

```
sebelasdecor-rag-antigravity/
│
├── knowledge/                    # 📚 Knowledge Base (sumber data RAG)
│   ├── pricelist.txt             #    5 paket dekorasi + harga
│   ├── katalog_tema.txt          #    5 tema dekorasi + deskripsi
│   └── faq.txt                   #    FAQ (DP, revisi, durasi, add-on)
│
├── chroma_db/                    # 💾 Vector Store (auto-generated)
│
├── venv/                         # 🐍 Python Virtual Environment
│
├── .env                          # 🔑 API Key & konfigurasi
├── rag_engine.py                 # 🧠 Core RAG Engine
├── app.py                        # 🌐 Flask API Server
├── index.html                    # 💬 Chat UI (WhatsApp-style)
└── requirements.txt              # 📦 Dependensi Python
```

---

## 🔧 Tech Stack

| Komponen | Teknologi | Keterangan |
|----------|-----------|------------|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Berjalan lokal di CPU, gratis |
| **Vector Database** | ChromaDB | Lokal, persistent di `chroma_db/` |
| **LLM (Generasi)** | 9router API (OpenAI-compatible) | Model: `FREE` (gemini-2.5-flash) |
| **Backend** | Python Flask + Flask-CORS | REST API `POST /api/chat` |
| **Frontend** | HTML + CSS + Vanilla JS | Dark theme WhatsApp-style |
| **Text Splitter** | LangChain `RecursiveCharacterTextSplitter` | Chunk: 500 chars, overlap: 80 |

---

## 🚀 Cara Menjalankan

### 1. Buat & Aktifkan Virtual Environment

```bash
# Buat venv
python -m venv venv

# Aktifkan (Git Bash)
source venv/Scripts/activate

# Aktifkan (PowerShell)
.\venv\Scripts\Activate.ps1

# Aktifkan (CMD)
venv\Scripts\activate.bat
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi API Key

Edit file `.env`:
```env
MIMO_API_KEY=sk-your-api-key-here
MIMO_BASE_URL=http://154.26.131.186:20128/v1
MIMO_MODEL=FREE
```

### 4. Jalankan Server

```bash
python app.py
```

Output:
```
🌸 Sebelas Decor — RAG Chatbot API Server
📡 Server berjalan di: http://127.0.0.1:5000
📮 Chat endpoint:      POST /api/chat
```

### 5. Buka Chat UI

Buka file `index.html` di browser.

---

## 📡 API Endpoint

### `POST /api/chat`

**Request:**
```json
{
  "message": "Berapa harga paket Indoor VIP?"
}
```

**Response:**
```json
{
  "reply": "Halo Kak! 😊 Paket Indoor VIP kami dihargai Rp 7.500.000..."
}
```

### `GET /`

Health check — menampilkan status server.

---

## 📚 Knowledge Base

### 1. Pricelist (`pricelist.txt`)
| Paket | Harga | Kapasitas |
|-------|-------|-----------|
| Indoor Basic | Rp 3.500.000 | 100 tamu |
| Indoor VIP | Rp 7.500.000 | 300 tamu |
| Outdoor Garden | Rp 6.000.000 | 150 tamu |
| Outdoor Beach | Rp 8.000.000 | 100 tamu |
| Akad Nikah | Rp 4.500.000 | 80 tamu |

### 2. Katalog Tema (`katalog_tema.txt`)
- 🪵 **Rustic Minimalist** — Natural, earth tone, kayu & burlap
- ✨ **Modern Elegant** — Kontemporer, akrilik, geometris
- 🏛️ **Traditional Jawa** — Gebyok, janur, batik
- 👑 **Glamour Gold** — Emas, kristal, chandelier
- 🌿 **Boho Chic** — Macrame, pampas grass, dream catcher

### 3. FAQ (`faq.txt`)
- DP minimal 50% untuk booking tanggal
- 2x revisi gratis (revisi ke-3+ dikenakan Rp 200.000)
- Durasi pengerjaan 2-5 hari kerja
- Add-on tersedia (photobooth, smoke machine, neon sign, dll.)
- Konsultasi tema GRATIS

---

## 🔄 Alur RAG (Retrieval-Augmented Generation)

```
1. User bertanya: "Berapa harga paket outdoor?"
                    │
2. Embedding       │  all-MiniLM-L6-v2 mengubah pertanyaan → vektor 384 dimensi
                    │
3. Similarity      │  ChromaDB mencari top-4 chunk paling relevan
   Search          │  berdasarkan cosine similarity
                    │
4. Context         │  Chunk teratas dijadikan KONTEKS untuk prompt LLM
   Building        │
                    │
5. LLM Generation  │  9router API (gemini-2.5-flash) merangkum konteks
                    │  menjadi jawaban ramah khas admin Sebelas Decor
                    │
6. Fallback        │  Jika skor relevansi < threshold (0.35),
                    │  bot menjawab sopan bahwa info tidak ditemukan
                    ▼
   Bot: "Halo Kak! Paket Outdoor Garden kami Rp 6.000.000..."
```

---

## 🔮 Roadmap & Integrasi Lanjutan

### WhatsApp Business (Meta Cloud API)
Arsitektur backend sudah siap diintegrasikan ke WhatsApp Business Cloud API:
- Tambahkan endpoint `/webhook` untuk menerima pesan WA dari Meta
- Fungsi `query_rag()` tetap sama — hanya perlu menambahkan handler webhook
- Gratis untuk pesan layanan pelanggan (customer-initiated, service messages)

### Potensi Pengembangan
- [ ] Integrasi WhatsApp Business API (Meta)
- [ ] Integrasi Telegram Bot
- [ ] Dashboard admin untuk update knowledge base
- [ ] Riwayat percakapan / session memory
- [ ] Analytics & reporting jumlah pertanyaan
- [ ] Deploy ke cloud (Railway / Render / VPS)

---

## 📄 Lisensi

Proyek internal **Sebelas Decor** — Hak cipta dilindungi.

---

<p align="center">
  🌸 Dibuat dengan ❤️ untuk <strong>Sebelas Decor</strong> 🌸
</p>
