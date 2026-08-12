"""
app.py — Flask Server untuk Sebelas Decor Chatbot
==================================================
Endpoint: POST /api/chat
Body: {"message": "pertanyaan user"}
Response: {"reply": "jawaban bot"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Rule-Based Engine (aktif) ──
from chat_engine import handle_chat

# ── RAG Engine (dinonaktifkan, simpan untuk pengembangan nanti) ──
# from rag_engine import query_rag

from db_client import save_lead, get_recent_leads, extract_lead_from_text

from flask import Flask, request, jsonify, send_from_directory
import os

# ──────────────────────────────────────────────
# Inisialisasi Flask App
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False
if hasattr(app, "json"):
    app.json.ensure_ascii = False

CORS(app)  # Izinkan CORS dari frontend


@app.after_request
def add_charset_header(response):
    """Pastikan respon JSON selalu menggunakan UTF-8 agar emoji tampil sempurna"""
    if response.mimetype == "application/json":
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


# ──────────────────────────────────────────────
# Pre-warm: DINONAKTIFKAN (tidak perlu untuk rule-based engine)
# Aktifkan kembali jika menggunakan RAG engine.
# ──────────────────────────────────────────────
# import threading
# def _warmup():
#     from rag_engine import _get_vectorstore
#     print("🔥 Pre-warming embedding model & vector store...")
#     _get_vectorstore()
#     print("✅ Pre-warm selesai! Server siap menerima request.")
#
# _warmup_thread = threading.Thread(target=_warmup, daemon=True)
# _warmup_thread.start()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Sebelas Decor Chatbot API & Database Service",
        "version": "2.0.0",
        "engine": "rule-based (lead collector)",
        "database": "PostgreSQL (Supabase)",
        "endpoints": {
            "chat": "POST /api/chat",
            "get_leads": "GET /api/leads",
            "create_lead": "POST /api/leads",
            "demo": "GET /demo atau /simulasi",
        }
    })


@app.route("/demo", methods=["GET"])
@app.route("/simulasi", methods=["GET"])
def simulation_page():
    """Halaman UI simulasi WhatsApp Chatbot Sebelas Decor"""
    return send_from_directory(os.path.dirname(__file__), "index.html")



@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint utama untuk chat.
    Menerima JSON:
      - message (str): pesan user
      - history (list): riwayat chat [{role, content}, ...]
      - customerName (str, optional): nama pelanggan (dari form / Meta Cloud API wa_name)
      - phone (str, optional): nomor HP (dari form / Meta Cloud API wa_id)
      - source (str, optional): "chatbot_web" | "whatsapp" (default: "chatbot_web")
    Mengembalikan JSON:
      - reply (str): balasan chatbot
      - leadSaved (bool): apakah lead baru berhasil disimpan
      - leadData (dict): data lead yang terkumpul sejauh ini
      - autoReply (bool): apakah bot harus auto-reply
      - handoverToAdmin (bool): apakah handover ke admin
    """
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "error": "Format tidak valid. Kirim JSON dengan key 'message'.",
                "example": {"message": "Halo, mau tanya pricelist dong"}
            }), 400

        user_message = data["message"].strip()
        history = data.get("history", [])
        phone_input = data.get("phone")            # Dari Meta Cloud API: wa_id
        source = data.get("source", "chatbot_web")

        if not user_message:
            return jsonify({
                "reply": "Halo! Ada yang bisa Sebelas Decor bantu? 😊",
                "autoReply": True,
                "handoverToAdmin": False,
                "leadSaved": False,
                "leadData": {},
            })

        # ── Rule-Based Engine ──
        result = handle_chat(
            user_message=user_message,
            history=history,
            phone=phone_input,
            source=source,
        )

        return jsonify(result)

        # ── RAG Engine (dinonaktifkan) ──
        # reply = query_rag(user_message, history=history, phone=phone_input)
        # return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ Error di /api/chat: {e}")
        return jsonify({
            "error": "Terjadi kesalahan internal. Silakan coba lagi.",
        }), 500


@app.route("/api/leads", methods=["GET"])
def get_leads_api():
    """Endpoint untuk mengambil lead terbaru dari database Supabase"""
    try:
        limit = request.args.get("limit", default=10, type=int)
        leads = get_recent_leads(limit=limit)
        return jsonify({"status": "success", "count": len(leads), "leads": leads})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads", methods=["POST"])
def create_lead_api():
    """Endpoint manual untuk memasukkan lead baru ke database Supabase"""
    try:
        data = request.get_json() or {}
        if not data.get("customerName"):
            return jsonify({"error": "customerName wajib diisi."}), 400

        result = save_lead(
            customer_name=data.get("customerName"),
            phone=data.get("phone"),
            event_date=data.get("eventDate"),
            location=data.get("location"),
            event_type=data.get("eventType", "Other"),
            package=data.get("package"),
            theme=data.get("theme"),
            status=data.get("status", "Inquiry"),
            notes=data.get("notes"),
            source=data.get("source", "chatbot_api")
        )
        return jsonify({"status": "success", "lead": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Jalankan Server
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌸 Sebelas Decor — Chatbot API Server (Rule-Based)")
    print("=" * 50)
    print("📡 Server berjalan di: http://127.0.0.1:5000")
    print("📮 Chat endpoint:      POST /api/chat")
    print("🔧 Engine:             Rule-Based Lead Collector")
    print("=" * 50 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
