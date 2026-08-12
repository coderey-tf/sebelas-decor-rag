"""
chat_engine.py — Rule-Based Lead Collector untuk Sebelas Decor Chatbot
======================================================================
Alur sederhana tanpa RAG/LLM:
1. Chat masuk → Kirim sapaan + form 4 data
2. User mengisi → Parsing regex (dari db_client.py)
3. Data lengkap → Simpan lead + kirim link pricelist GDrive
4. Selesai → Handover ke admin manusia
"""

import time
from db_client import extract_lead_from_conversation, save_lead, get_lead_by_phone

# ──────────────────────────────────────────────
# Link Pricelist Google Drive per Kategori
# (Untuk saat ini pakai 1 link yang sama.
#  Nanti bisa diganti per kategori jika sudah ada.)
# ──────────────────────────────────────────────
PRICELIST_LINKS = {
    "Wedding Gedung": "https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link",
    "Wedding Rumah": "https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link",
    "Engagement Gedung": "https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link",
    "Engagement Rumah": "https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link",
}

DEFAULT_PRICELIST_LINK = "https://drive.google.com/file/d/1TKXd4R10wQFI_BL9_Z4nD8iXiXsD9k7X/view?usp=drive_link"

# ──────────────────────────────────────────────
# Pesan-Pesan Template
# ──────────────────────────────────────────────
GREETING_MESSAGE = (
    "Halo Kak! 👋 Selamat datang di Sebelas Decor. ✨\n\n"
    "Senang sekali bisa menyambut Kakak! Biar kami bisa memberikan rekomendasi katalog & pricelist yang paling pas untuk acara impian Kakak, boleh dibantu infokan detail rencananya:\n\n"
    "1. 👤 **Nama** : \n"
    "2. 📅 **Rencana tanggal acara** : (contoh: 20 Oktober 2026)\n"
    "3. 💒 **Jenis acara (Pernikahan / Lamaran)** : \n"
    "4. 🏛️ **Rencana tempat (Gedung / Rumah)** : \n\n"
    "💡 *Catatan: Jika tanggal acaranya belum ada, tidak apa-apa dikosongi atau dilewati dulu ya Kak!*\n\n"
    "Ditunggu informasinya ya Kak! 😊"
)


def _build_followup_message(lead_info: dict) -> str:
    """Bangun pesan follow-up yang menanyakan field yang masih kosong."""
    missing = []

    if not lead_info.get("customer_name"):
        missing.append("👤 **Nama Kakak**")
    if not lead_info.get("event_type"):
        missing.append("💒 **Jenis acara** (Pernikahan atau Lamaran)")
    if not lead_info.get("venue_type"):
        missing.append("🏛️ **Rencana tempat** (Gedung atau Rumah)")

    if not missing:
        return ""

    items = "\n".join(f"- {m}" for m in missing)
    return (
        f"Terima kasih infonya Kak! 😊\n\n"
        f"Boleh dilengkapi lagi ya Kak untuk data berikut:\n{items}\n\n"
        f"💡 *Untuk tanggal acara, boleh dikosongi dulu jika belum pasti ya Kak!*"
    )


def _build_pricelist_message(lead_info: dict) -> str:
    """Bangun pesan pricelist + handover setelah data lengkap."""
    name = lead_info.get("customer_name", "Kak")
    pkg = lead_info.get("package", "Pilihan")
    link = PRICELIST_LINKS.get(pkg, DEFAULT_PRICELIST_LINK)

    return (
        f"Terima kasih banyak, Kak {name}! ✨\n\n"
        f"Berikut link Pricelist **{pkg}** resmi Sebelas Decor yang sesuai dengan kebutuhan Kakak:\n\n"
        f"📄 **Pricelist {pkg} Sebelas Decor**:\n"
        f"{link}\n\n"
        f"Silakan dipelajari rincian paketnya ya Kak. Data Kakak sudah kami catat, dan percakapan ini akan segera dilanjutkan langsung oleh Admin kami untuk konsultasi & penyesuaian lebih lanjut! 😊"
    )


def _is_greeting(message: str) -> bool:
    """Cek apakah pesan hanya sapaan singkat tanpa data event."""
    msg_lower = message.lower().strip()

    greeting_words = [
        'halo', 'hi', 'hello', 'hay', 'hey', 'pagi', 'siang', 'sore',
        'malam', 'selamat', 'permisi', 'assalam', 'spill', 'min', 'kak',
        'p', 'hai',
    ]
    event_keywords = [
        'wedding', 'nikah', 'lamaran', 'engagement', 'tunangan',
        'gedung', 'rumah', 'hotel', 'hall', 'ballroom',
        'januari', 'februari', 'maret', 'april', 'mei', 'juni',
        'juli', 'agustus', 'september', 'oktober', 'november', 'desember',
    ]

    has_greeting = any(gw in msg_lower for gw in greeting_words)
    has_event = any(kw in msg_lower for kw in event_keywords)

    import re
    has_digits = bool(re.search(r'\d', message))

    return has_greeting and not has_event and not has_digits and len(msg_lower) <= 60


# ──────────────────────────────────────────────
# Fungsi Utama: handle_chat
# ──────────────────────────────────────────────
def handle_chat(user_message: str, history: list = None, phone: str = None, source: str = "chatbot_web") -> dict:
    """
    Entry point utama chatbot rule-based.

    Returns dict:
        reply (str): pesan balasan
        leadSaved (bool): apakah lead baru disimpan
        leadData (dict): data lead terkumpul
        autoReply (bool): apakah bot harus auto-reply
        handoverToAdmin (bool): apakah handover ke admin
    """
    if not user_message or not user_message.strip():
        return {
            "reply": "Halo! Ada yang bisa Sebelas Decor bantu? 😊",
            "leadSaved": False,
            "leadData": {},
            "autoReply": True,
            "handoverToAdmin": False,
        }

    # ── 1. Cek apakah lead sudah ada di database ──
    if phone:
        existing_lead = get_lead_by_phone(phone)
        if existing_lead:
            return {
                "reply": "",
                "autoReply": False,
                "handoverToAdmin": True,
                "leadSaved": False,
                "leadData": existing_lead,
            }

    # ── 2. Jika sapaan murni (tanpa data) → Kirim greeting + form ──
    if _is_greeting(user_message) and (not history or len(history) <= 1):
        time.sleep(1.2)  # Jeda alami agar terasa manusiawi
        return {
            "reply": GREETING_MESSAGE,
            "leadSaved": False,
            "leadData": {},
            "autoReply": True,
            "handoverToAdmin": False,
        }

    # ── 3. Ekstrak data lead dari percakapan ──
    lead_info = extract_lead_from_conversation(
        current_message=user_message,
        history=history,
        wa_phone=phone,
    )

    # ── 4. Data lengkap → Simpan lead + kirim pricelist + handover ──
    if lead_info.get("is_complete"):
        name = lead_info.get("customer_name")
        if not name:
            phone_display = lead_info.get("phone", "Anonim")
            name = f"Pelanggan Chatbot ({phone_display})"

        lead_saved = False
        result = save_lead(
            customer_name=name,
            phone=lead_info.get("phone"),
            event_date=lead_info.get("event_date"),
            location=lead_info.get("location"),
            event_type=lead_info.get("event_type", "Other"),
            package=lead_info.get("package"),
            status="Inquiry",
            notes=f"Auto-captured via {source}. Filter: {lead_info.get('package', '-')}, Tanggal: {lead_info.get('event_date', '-')}",
            source=source,
        )
        if "error" not in result:
            lead_saved = True

        time.sleep(1.0)
        reply = _build_pricelist_message(lead_info)

        return {
            "reply": reply,
            "leadSaved": lead_saved,
            "leadData": {
                "customer_name": lead_info.get("customer_name"),
                "event_type": lead_info.get("event_type"),
                "venue_type": lead_info.get("venue_type"),
                "event_date": lead_info.get("event_date"),
                "package": lead_info.get("package"),
                "location": lead_info.get("location"),
                "is_complete": True,
            },
            "autoReply": True,
            "handoverToAdmin": True,
        }

    # ── 5. Data belum lengkap → Tanyakan field yang kosong ──
    followup = _build_followup_message(lead_info)
    if not followup:
        # Fallback: Kirim greeting lagi
        followup = GREETING_MESSAGE

    time.sleep(1.0)
    return {
        "reply": followup,
        "leadSaved": False,
        "leadData": {
            "customer_name": lead_info.get("customer_name"),
            "event_type": lead_info.get("event_type"),
            "venue_type": lead_info.get("venue_type"),
            "event_date": lead_info.get("event_date"),
            "package": lead_info.get("package"),
            "location": lead_info.get("location"),
            "is_complete": False,
        },
        "autoReply": True,
        "handoverToAdmin": False,
    }
