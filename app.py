"""
app.py — Flask Server untuk Sebelas Decor RAG Chatbot
=====================================================
Endpoint: POST /api/chat
Body: {"message": "pertanyaan user"}
Response: {"reply": "jawaban bot"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import query_rag
from db_client import save_lead, get_recent_leads, extract_lead_from_text

from flask import Flask, request, jsonify, send_from_directory
import os
import threading

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
# Pre-warm: Load embedding model + vectorstore di background thread
# agar request pertama user tidak menunggu 16-30 detik.
# ──────────────────────────────────────────────
def _warmup():
    from rag_engine import _get_vectorstore
    print("🔥 Pre-warming embedding model & vector store...")
    _get_vectorstore()
    print("✅ Pre-warm selesai! Server siap menerima request.")

_warmup_thread = threading.Thread(target=_warmup, daemon=True)
_warmup_thread.start()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Sebelas Decor RAG Chatbot API & Database Service",
        "version": "1.1.0",
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
    """
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({
                "error": "Format tidak valid. Kirim JSON dengan key 'message'.",
                "example": {"message": "Berapa harga paket Indoor Basic?"}
            }), 400

        user_message = data["message"].strip()
        history = data.get("history", [])
        customer_name = data.get("customerName")  # Dari Meta Cloud API: wa_profile_name
        phone_input = data.get("phone")            # Dari Meta Cloud API: wa_id
        source = data.get("source", "chatbot_web")

        if not user_message:
            return jsonify({
                "reply": "Halo! Ada yang bisa Sebelas Decor bantu? 😊"
            })

        # Panggil RAG engine dengan history
        reply = query_rag(user_message, history=history)

        # ── Akumulasi data lead dari seluruh percakapan ──
        from db_client import extract_lead_from_conversation
        lead_info = extract_lead_from_conversation(
            current_message=user_message,
            history=history,
            wa_name=customer_name,
            wa_phone=phone_input,
        )

        lead_saved = False

        # Auto-insert lead ke database saat 3 filter lengkap
        if lead_info.get("is_complete"):
            # Tentukan nama pelanggan
            name = lead_info.get("customer_name")
            if not name:
                phone_display = lead_info.get("phone", "Anonim")
                name = f"Pelanggan Chatbot ({phone_display})"

            result = save_lead(
                customer_name=name,
                phone=lead_info.get("phone"),
                event_date=lead_info.get("event_date"),
                location=lead_info.get("location"),
                event_type=lead_info.get("event_type", "Other"),
                package=lead_info.get("package"),  # e.g. "Wedding Gedung"
                status="Inquiry",
                notes=f"Auto-captured via {source}. Filter: {lead_info.get('package', '-')}, Tanggal: {lead_info.get('event_date', '-')}",
                source=source,
            )
            if "error" not in result:
                lead_saved = True

        return jsonify({
            "reply": reply,
            "leadSaved": lead_saved,
            "leadData": {
                "event_type": lead_info.get("event_type"),
                "venue_type": lead_info.get("venue_type"),
                "event_date": lead_info.get("event_date"),
                "package": lead_info.get("package"),
                "location": lead_info.get("location"),
                "is_complete": lead_info.get("is_complete", False),
            }
        })

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
    print("🌸 Sebelas Decor — RAG Chatbot API Server")
    print("=" * 50)
    print("📡 Server berjalan di: http://127.0.0.1:5000")
    print("📮 Chat endpoint:      POST /api/chat")
    print("=" * 50 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
