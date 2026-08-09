"""
rag_engine.py — RAG Engine untuk Sebelas Decor Chatbot
=====================================================
Dense Vector Retrieval menggunakan:
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2) — lokal & gratis
- ChromaDB sebagai vector store lokal
- MiMo API sebagai LLM untuk generasi jawaban
"""

import os
import sys
import glob
import json
import re
import requests
import warnings
from dotenv import load_dotenv

# Reconfigure stdout ke UTF-8 untuk kompatibilitas Windows console print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Redam warning deprecation bawaan LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ──────────────────────────────────────────────
# 1. Load environment variables
# ──────────────────────────────────────────────
load_dotenv()
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "http://154.26.131.186:20128/v1").rstrip("/")
MIMO_API_URL = f"{MIMO_BASE_URL}/chat/completions"
MIMO_MODEL = os.getenv("MIMO_MODEL", "FREE")

# ──────────────────────────────────────────────
# 2. Konfigurasi
# ──────────────────────────────────────────────
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 6
RELEVANCE_THRESHOLD = 0.15  # skor jarak / similarity threshold


# ──────────────────────────────────────────────
# 3. Lazy-loaded Embedding Model & Vector Store
#    (hanya di-init sekali saat pertama dipakai)
# ──────────────────────────────────────────────
_embeddings = None
_vectorstore = None


def _get_embeddings():
    """Lazy-load embedding model. Dipanggil sekali, lalu di-cache."""
    global _embeddings
    if _embeddings is None:
        print("⏳ Memuat embedding model (hanya sekali)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Embedding model siap!")
    return _embeddings


def _get_vectorstore():
    """Lazy-load vector store. Pakai yang sudah ada di disk, atau build baru."""
    global _vectorstore
    if _vectorstore is None:
        emb = _get_embeddings()
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            print("📂 Memuat vector store dari disk...")
            _vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=emb,
            )
        else:
            print("🔨 Membangun vector store baru dari knowledge base...")
            docs = load_documents()
            if not docs:
                raise FileNotFoundError("Tidak ada dokumen di folder knowledge/")
            chunks = split_documents(docs)
            _vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=emb,
                persist_directory=CHROMA_DIR,
            )
            print(f"💾 Vector store tersimpan di: {CHROMA_DIR}")
    return _vectorstore


def load_documents():
    """Membaca semua file .txt di folder knowledge/"""
    docs = []
    txt_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))

    if not txt_files:
        print(f"⚠️  Tidak ada file .txt di {KNOWLEDGE_DIR}")
        return docs

    for fpath in txt_files:
        try:
            loader = TextLoader(fpath, encoding="utf-8")
            docs.extend(loader.load())
            print(f"   📄 Loaded: {os.path.basename(fpath)}")
        except Exception as e:
            print(f"   ❌ Gagal load {fpath}: {e}")

    return docs


def split_documents(docs):
    """Memecah dokumen menjadi chunk-chunk kecil"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "---", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Total chunks: {len(chunks)}")
    return chunks


def rebuild_vectorstore():
    """Rebuild vector store secara aman (untuk admin/manual refresh)."""
    global _vectorstore
    emb = _get_embeddings()
    docs = load_documents()
    if not docs:
        raise FileNotFoundError("Tidak ada dokumen di folder knowledge/")
    chunks = split_documents(docs)

    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    try:
        col_data = vs._collection.get()
        if col_data and "ids" in col_data and col_data["ids"]:
            vs._collection.delete(ids=col_data["ids"])
            print(f"🗑️ Membersihkan {len(col_data['ids'])} chunk lama di vector store.")
    except Exception as e:
        print(f"⚠️ Warning membersihkan collection lama: {e}")

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=CHROMA_DIR,
    )
    print(f"💾 Vector store berhasil di-rebuild ({_vectorstore._collection.count()} chunks) di: {CHROMA_DIR}")
    return _vectorstore



# ──────────────────────────────────────────────
# 5. Semantic Search
# ──────────────────────────────────────────────
def semantic_search(query, k=TOP_K):
    """
    Melakukan similarity search ke ChromaDB.
    Mengembalikan list of (Document, score).
    Score = jarak (distance) — makin kecil makin relevan.
    """
    results = _get_vectorstore().similarity_search_with_relevance_scores(query, k=k)
    return results


# ──────────────────────────────────────────────
# 6. Panggil MiMo API
# ──────────────────────────────────────────────
def call_mimo_api(system_prompt, user_prompt, history=None):
    """Mengirim prompt ke MiMo API dan mengembalikan respons teks (dengan support history)"""
    if not MIMO_API_KEY or MIMO_API_KEY == "your_mimo_api_key_here":
        return (
            "⚠️ API Key MiMo belum dikonfigurasi. "
            "Silakan isi MIMO_API_KEY di file .env dengan API key yang valid."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIMO_API_KEY}",
    }

    messages = [{"role": "system", "content": system_prompt}]

    # Tambahkan percakapan terdahulu dari history
    if history and isinstance(history, list):
        for msg in history[-6:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MIMO_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 450,
        "stream": True,
    }


    try:
        # Gunakan stream=True agar bisa membaca SSE line-by-line
        response = requests.post(
            MIMO_API_URL, json=payload, headers=headers,
            timeout=60, stream=True
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        # ── Kasus 1: Server mengirim SSE stream (text/event-stream) ──
        if "text/event-stream" in content_type or "chunked" in response.headers.get("Transfer-Encoding", ""):
            return _parse_sse_stream(response)

        # ── Kasus 2: Baca seluruh body ──
        raw_text = response.text.strip()
        if not raw_text:
            return "❌ Server API mengembalikan respon kosong."

        # Cek apakah body ternyata tetap SSE (data: ...)
        if raw_text.startswith("data:"):
            return _parse_sse_text(raw_text)

        # ── Kasus 3: JSON standar ──
        return _parse_json_response(raw_text)

    except requests.exceptions.Timeout:
        return "⏱️ Maaf, koneksi ke server AI timeout (60s). Silakan coba lagi."
    except requests.exceptions.ConnectionError:
        return "🔌 Maaf, tidak dapat terhubung ke server AI. Pastikan server dapat diakses."
    except requests.exceptions.HTTPError as e:
        return f"❌ Error dari API ({e.response.status_code}): {e.response.text[:250]}"
    except Exception as e:
        return f"❌ Terjadi kesalahan: {str(e)}"


def _parse_sse_stream(response):
    """
    Parse response stream. Mendukung 2 format:
    1. SSE sesungguhnya (tiap baris diawali 'data: {...}')
    2. JSON biasa yang dikirim via chunked transfer encoding
    """
    collected_sse = []
    raw_lines = []

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        raw_lines.append(line)

        # Coba parse sebagai SSE (data: ...)
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if json_str == "[DONE]":
                break
            if not json_str:
                continue
            try:
                chunk = json.loads(json_str)
                text_piece = _extract_delta(chunk)
                if text_piece:
                    collected_sse.append(text_piece)
            except Exception:
                pass

    # Kasus 1: Berhasil parse sebagai SSE stream
    if collected_sse:
        return "".join(collected_sse).strip()

    # Kasus 2: Bukan SSE — gabungkan semua baris dan coba parse sebagai JSON biasa
    if raw_lines:
        full_text = "".join(raw_lines)
        return _parse_json_response(full_text)


def _parse_sse_text(raw_text):
    """Parse SSE dari raw text string (bukan streaming iterator)"""
    collected = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        json_str = line[5:].strip()
        if json_str == "[DONE]":
            break
        if not json_str:
            continue
        try:
            chunk = json.loads(json_str)
            text_piece = _extract_delta(chunk)
            if text_piece:
                collected.append(text_piece)
        except Exception:
            pass

    if collected:
        return "".join(collected).strip()
    return f"❌ Format SSE tidak mengandung teks: {raw_text[:200]}"


def _extract_delta(chunk):
    """Ekstrak teks dari satu chunk SSE (format OpenAI delta atau message)"""
    if not isinstance(chunk, dict) or "choices" not in chunk:
        return ""
    choices = chunk["choices"]
    if not choices:
        return ""
    c = choices[0]
    if not isinstance(c, dict):
        return ""
    # Stream format: delta.content
    delta = c.get("delta", {})
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])
    # Non-stream format: message.content
    msg = c.get("message", {})
    if isinstance(msg, dict) and msg.get("content"):
        return str(msg["content"])
    # Fallback: text field
    if c.get("text"):
        return str(c["text"])
    return ""


def _parse_json_response(raw_text):
    """Parse respons JSON standar (non-streaming)"""
    try:
        data = json.loads(raw_text)
    except Exception:
        return f"❌ Respon bukan JSON valid: {raw_text[:200]}"

    if isinstance(data, dict):
        # OpenAI standard format
        if "choices" in data and data["choices"]:
            text = _extract_delta(data)
            if text:
                return text.strip()
        if "response" in data:
            return str(data["response"]).strip()
        if "content" in data:
            return str(data["content"]).strip()
        if "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                return f"❌ Error API: {err.get('message', err)}"
            return f"❌ Error API: {err}"

    return f"❌ Format respon tidak dikenali: {raw_text[:200]}"




# ──────────────────────────────────────────────
# 7. Fungsi Utama: query_rag
# ──────────────────────────────────────────────
def query_rag(user_message, history=None):
    """
    Pipeline utama RAG dengan alur sales percakapan & multi-turn memory:
    """
    if not user_message or not user_message.strip():
        return "Halo! Ada yang bisa Sebelas Decor bantu? 😊"

    # --- Fast-Path: Respon Instan untuk Sapaan Sederhana ---
    import re
    msg_clean = re.sub(r'[^\w\s]', '', user_message.lower()).strip()
    greetings = {
        "halo", "halo kak", "hi", "hi kak", "hello", "p", "ping", "spill",
        "selamat siang", "selamat pagi", "selamat sore", "selamat malam",
        "pagi", "siang", "sore", "malam", "siang kak", "pagi kak", "sore kak", "malam kak",
        "permisi", "assalamualaikum", "assalamu alaikum", "min", "halo min",
        "selamat siang kak", "selamat pagi kak", "selamat sore kak", "selamat malam kak"
    }

    greeting_pattern = r'^(halo|hi|hello|p|ping|spill|selamat\s+(pagi|siang|sore|malam)|pagi|siang|sore|malam|permisi|assalamualaikum|assalamu\s+alaikum)(\s+(kak|min|gan|sis|admin|sebelas\s+decor))?$'
    is_greeting_msg = bool(re.match(greeting_pattern, msg_clean))

    if is_greeting_msg or msg_clean in greetings or (len(msg_clean) <= 20 and any(g in msg_clean for g in ["halo", "hi", "pagi", "siang", "sore", "malam", "permisi", "assalam"])):
        from db_client import extract_lead_from_conversation
        lead_info = extract_lead_from_conversation(user_message, history=history)
        if not lead_info.get("is_complete"):
            return (
                "Halo kak! 👋 Selamat datang di Sebelas Decor. ✨\n\n"
                "Senang sekali bisa membantu mewujudkan dekorasi impian Kakak!\n"
                "Untuk memberikan rincian katalog & pricelist yang paling sesuai, boleh dibantu infokan 3 detail berikut Kak?\n\n"
                "1) 📅 **Tanggal berapa** rencana acaranya?\n"
                "2) 💒 Acaranya untuk **Pernikahan (Wedding)** atau **Lamaran (Engagement)**?\n"
                "3) 🏛️ Lokasinya di **Gedung/Hotel** atau di **Rumah**?\n\n"
                "Setelah 3 detail ini lengkap, kami akan langsung kirimkan pricelist spesifiknya ya! 😊"
            )

    # --- Semantic Search ---
    results = semantic_search(user_message)

    # --- Debug: tampilkan skor relevansi di terminal ---
    print(f"\n🔍 Query: '{user_message}'")
    if results:
        for i, (doc, score) in enumerate(results):
            source = os.path.basename(doc.metadata.get("source", "?"))
            print(f"   [{i+1}] Skor: {score:.4f} | Sumber: {source} | Preview: {doc.page_content[:80]}...")
    else:
        print("   ⚠️ Tidak ada hasil dari ChromaDB!")

    # Filter hasil berdasarkan threshold relevansi
    relevant_results = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

    print(f"   ✅ {len(relevant_results)}/{len(results)} chunk lolos threshold ({RELEVANCE_THRESHOLD})")

    if not relevant_results and results:
        print("   ⚠️ Semua skor di bawah threshold, menggunakan top result sebagai fallback")
        relevant_results = [results[0]]

    # --- Bangun context dari top-k chunks ---
    context_parts = []
    if relevant_results:
        for i, (doc, score) in enumerate(relevant_results, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            context_parts.append(
                f"[Sumber: {source} | Relevansi: {score:.2f}]\n{doc.page_content}"
            )

    context = "\n\n---\n\n".join(context_parts) if context_parts else "Data pricelist, katalog tema, dan FAQ Sebelas Decor."

    # --- Cek Ketersediaan Tanggal di Database Supabase ---
    availability_context = ""
    # Cek dari pesan user atau history percakapan
    search_text = user_message
    if history and isinstance(history, list):
        for msg in reversed(history[-4:]):
            if isinstance(msg, dict) and msg.get("content"):
                search_text += " " + msg["content"]

    from db_client import check_date_availability, parse_indonesian_date
    parsed_date = parse_indonesian_date(search_text)
    
    if parsed_date:
        avail_res = check_date_availability(parsed_date)
        if avail_res.get("available") is False:
            bookings_info = avail_res.get("existing_bookings", [])
            booking_details = ", ".join([f"{b.get('event_type')} ({b.get('package', 'Pilihan')})" for b in bookings_info])
            availability_context = f"""
⚠️ STATUS KETERSEDIAAN TANGGAL DI DATABASE SUPABASE:
- Tanggal {avail_res.get('date_formatted')} SUDAH TERISI / PENUH (Status: Booked/DP Paid di Database).
- Detail Booking Eksisting: {booking_details}

INSTRUKSI KHUSUS KETERSEDIAAN:
Kamu HARUS memberi tahu pelanggan dengan sangat sopan bahwa tanggal {avail_res.get('date_formatted')} SUDAH PENUH/TERISI (sudah ada booking masuk). Mohon minta maaf dan tawarkan pelanggan opsi tanggal alternatif lain atau jadwal terdekat! DILARANG KERAS mengatakan tanggal ini masih tersedia.
"""
        elif avail_res.get("available") is True and avail_res.get("date_formatted"):
            availability_context = f"""
✅ STATUS KETERSEDIAAN TANGGAL DI DATABASE SUPABASE:
- Tanggal {avail_res.get('date_formatted')} MASIH TERSEDIA (Kosong di Database).

INSTRUKSI KHUSUS KETERSEDIAAN:
Konfirmasikan dengan gembira bahwa tanggal {avail_res.get('date_formatted')} masih tersedia untuk dipesan!
"""

    # --- System Prompt dengan Alur 3-Filter Sales Funnel ---
    system_prompt = f"""Kamu adalah asisten virtual resmi & admin sales utama dari "Sebelas Decor" (jasa dekorasi event: pernikahan/wedding & lamaran/engagement).

SISTEM 3-FILTER SALES FUNNEL:
Sebelas Decor memiliki 4 KATEGORI PAKET yang masing-masing punya pricelist tersendiri:
1. 💒 **Wedding Gedung** — Pernikahan di Gedung/Ballroom/Hotel/Hall/Masjid
2. 🏡 **Wedding Rumah** — Pernikahan di Rumah/Halaman/Garasi/Outdoor Kediaman
3. 💍 **Engagement Gedung** — Lamaran di Gedung/Resto/Hotel Function Room
4. 🏠 **Engagement Rumah** — Lamaran di Rumah/Kediaman

ALUR INTERAKSI PERCAKAPAN (WAJIB DIIKUTI SECARA BERURUTAN):

TAHAP 1 — PENGUMPULAN DATA (JANGAN KIRIM PRICELIST DULU!):
   Jika klien baru menyapa, bertanya tentang dekorasi, atau meminta pricelist:
   - Sambut dengan hangat dan ramah.
   - Sampaikan bahwa kamu perlu mengetahui beberapa detail dulu agar bisa memberikan pricelist yang paling sesuai.
   - JANGAN berikan link pricelist sebelum 3 detail berikut terkumpul! Tanyakan:
     1) 📅 **Tanggal berapa** rencana acaranya?
     2) 💒 Acaranya untuk **Pernikahan (Wedding)** atau **Lamaran (Engagement)**?
     3) 🏛️ Lokasi tempat acaranya di **Gedung/Hotel** atau di **Rumah**?
   - Jika klien hanya menjawab sebagian, tanyakan detail yang belum dijawab dengan sopan.

TAHAP 2 — KIRIM PRICELIST SPESIFIK (SETELAH 3 DATA LENGKAP):
   Setelah ketiga detail di atas sudah terkumpul dari percakapan:
   - Perhatikan info ketersediaan tanggal dari database di bawah. Jika tanggal SUDAH PENUH, sampaikan maaf dan tawarkan opsi tanggal lain. Jika MASIH TERSEDIA, konfirmasikan dengan gembira.
   - Kirim link Google Drive PDF Pricelist yang SESUAI KATEGORI klien:
     📄 **Pricelist [KATEGORI] Sebelas Decor**:
     https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link
     (Ganti [KATEGORI] dengan: Wedding Gedung / Wedding Rumah / Engagement Gedung / Engagement Rumah sesuai jawaban klien.)
   - WAJIB TAMPILKAN BROSUR PROMO SESUAI ACARA KLIEN:
     • Untuk WEDDING / PERNIKAHAN:
       ![Promo Wedding](/static/images/promo_wedding.png)
     • Untuk ENGAGEMENT / LAMARAN:
       ![Promo Engagement](/static/images/promo_engagement.png)
   - Tawarkan konsultasi desain & penyesuaian tema gratis.

{availability_context}

ATURAN PENTING:
- DILARANG memberikan link pricelist sebelum 3 detail (tanggal, jenis acara, tipe venue) terkumpul!
- DILARANG KERAS menyarankan hubungi admin via WhatsApp karena KAMU ADALAH admin utama di chat ini!
- Gunakan bahasa Indonesia yang sopan, ramah, dan komunikatif.
- Jawab dengan singkat dan tidak bertele-tele (maksimal 150 kata)."""


    # --- User Prompt ---
    user_prompt = f"""KONTEKS KNOWLEDGE BASE SEBELAS DECOR:
{context}

===

PERTANYAAN PELANGGAN:
{user_message}

INSTRUKSI: Jawab pelanggan sebagai admin Sebelas Decor. Ikuti alur 3-Filter Sales Funnel secara berurutan. JANGAN kirim pricelist sebelum 3 detail (tanggal, jenis acara, tipe venue) terkumpul!"""

    # --- Panggil MiMo API dengan history ---
    answer = call_mimo_api(system_prompt, user_prompt, history=history)

    return answer


# ──────────────────────────────────────────────
# 8. CLI Test (opsional)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌸 Sebelas Decor — RAG Chatbot (CLI Mode)")
    print("=" * 50)
    print("Ketik pertanyaan Anda, atau 'quit' untuk keluar.\n")

    while True:
        try:
            user_input = input("👤 Anda: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Terima kasih! Sampai jumpa lagi.")
                break
            if not user_input:
                continue

            print("🤖 Sebelas Decor:", end=" ")
            response = query_rag(user_input)
            print(response)
            print()
        except KeyboardInterrupt:
            print("\n👋 Terima kasih! Sampai jumpa lagi.")
            break
