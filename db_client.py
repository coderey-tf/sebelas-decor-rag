"""
db_client.py — Database Client untuk Sebelas Decor RAG Chatbot
=============================================================
Menghubungkan RAG Chatbot ke PostgreSQL Database (Supabase)
agar lead baru yang didapatkan AI chatbot langsung tersimpan
ke database dan dapat dilihat di Sebelas Decor Dashboard secara real-time.
"""

import os
import sys
import uuid
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Gunakan DIRECT_URL jika ada (port 5432) atau fallback ke DATABASE_URL (tanpa query param pgbouncer jika direct)
RAW_DB_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL", "")

def get_db_connection():
    """Membuat koneksi ke database PostgreSQL Supabase"""
    if not RAW_DB_URL:
        raise ValueError("DATABASE_URL atau DIRECT_URL tidak ditemukan di .env")
    
    # Hapus parameter pgbouncer=true jika menghubungkan lewat psycopg2 biasa
    db_url = RAW_DB_URL.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
    return psycopg2.connect(db_url)


def get_lead_by_phone(phone: str) -> dict:
    """Mengambil data lead berdasarkan nomor HP (unique key WhatsApp) jika sudah ada."""
    if not phone:
        return None
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, customer_name, phone, event_date, location, event_type, package, status, notes, created_at
            FROM leads
            WHERE phone = %s;
        """, (phone,))
        lead = cursor.fetchone()
        return dict(lead) if lead else None
    except Exception as e:
        print(f"❌ [DB Error get_lead_by_phone]: {e}")
        return None
    finally:
        if conn:
            conn.close()


def save_lead(
    customer_name: str,
    phone: str = None,
    event_date: str = None,
    location: str = None,
    event_type: str = "Other", # Wedding, Engagement, Birthday, Other
    package: str = None,
    theme: str = None,
    status: str = "Inquiry",
    notes: str = None,
    source: str = "chatbot"
) -> dict:
    """
    Menyimpan atau meng-update (UPSERT) lead berdasarkan nomor HP (phone)
    agar tidak terjadi duplicate leads di database Supabase.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        now = datetime.utcnow()
        
        # Validasi & Mapping Enum LeadStatus untuk PostgreSQL
        STATUS_MAP = {
            "Inquiry": "Inquiry",
            "FollowUp": "Follow-up",
            "Follow-up": "Follow-up",
            "Booked": "Booked",
            "DpPaid": "DP Paid",
            "DP Paid": "DP Paid",
            "Completed": "Completed",
            "Cancelled": "Cancelled"
        }
        db_status = STATUS_MAP.get(status, "Inquiry")

        # Validasi enum EventType
        valid_event_types = ["Wedding", "Engagement", "Birthday", "Other"]
        if event_type not in valid_event_types:
            event_type = "Other"
            
        # Parse tanggal jika string
        parsed_date = None
        if event_date:
            try:
                if isinstance(event_date, str):
                    parsed_date = datetime.strptime(event_date[:10], "%Y-%m-%d").date()
                elif isinstance(event_date, datetime):
                    parsed_date = event_date.date()
            except Exception:
                parsed_date = None

        # ── Cek apakah lead dengan nomor HP ini sudah ada ──
        existing_lead = None
        if phone:
            cursor.execute("SELECT id, status, customer_name FROM leads WHERE phone = %s;", (phone,))
            existing_lead = cursor.fetchone()

        if existing_lead:
            # ── UPDATE (Mencegah Double Lead) ──
            lead_id = existing_lead["id"]
            update_query = """
            UPDATE leads SET
                customer_name = COALESCE(%s, customer_name),
                event_date = COALESCE(%s, event_date),
                location = COALESCE(%s, location),
                event_type = %s::"EventType",
                package = COALESCE(%s, package),
                theme = COALESCE(%s, theme),
                notes = COALESCE(%s, notes),
                source = %s,
                updated_at = %s
            WHERE id = %s
            RETURNING id, customer_name, phone, event_type, status, created_at;
            """
            cursor.execute(update_query, (
                customer_name,
                parsed_date,
                location,
                event_type,
                package,
                theme,
                notes,
                source,
                now,
                lead_id
            ))
            updated_lead = cursor.fetchone()

            # Activity Log Update
            activity_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO activity_logs (id, lead_id, action, details, performed_by, created_at)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                activity_id,
                lead_id,
                "Lead Updated",
                f"Data lead di-update otomatis oleh Chatbot AI (Paket: {package or '-'})",
                "chatbot",
                now
            ))
            conn.commit()
            print(f"🔄 [DB] Lead di-update (Upsert): {customer_name or existing_lead['customer_name']} (ID: {lead_id})")
            return dict(updated_lead)

        else:
            # ── INSERT Lead Baru ──
            lead_id = str(uuid.uuid4())
            insert_lead_query = """
            INSERT INTO leads (
                id, customer_name, phone, event_date, location, event_type,
                package, theme, status, notes, source, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s::"EventType", %s, %s, %s::"LeadStatus", %s, %s, %s, %s
            )
            RETURNING id, customer_name, phone, event_type, status, created_at;
            """
            cursor.execute(insert_lead_query, (
                lead_id,
                customer_name,
                phone,
                parsed_date,
                location,
                event_type,
                package,
                theme,
                db_status,
                notes,
                source,
                now,
                now
            ))
            new_lead = cursor.fetchone()

            # Activity Log Insert
            activity_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO activity_logs (id, lead_id, action, details, performed_by, created_at)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                activity_id,
                lead_id,
                "Lead Created",
                f"Lead baru berhasil ditangkap otomatis oleh Chatbot AI (Sumber: {source})",
                "chatbot",
                now
            ))
            conn.commit()
            print(f"✅ [DB] Lead baru disimpan: {customer_name} (ID: {lead_id})")
            return dict(new_lead)
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ [DB] Error menyimpan lead: {e}")
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


def get_recent_leads(limit=10):
    """Mengambil daftar lead terbaru dari database"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, customer_name, phone, event_type, status, location, package, theme, created_at
            FROM leads
            ORDER BY created_at DESC
            LIMIT %s;
        """, (limit,))
        leads = cursor.fetchall()
        return [dict(row) for row in leads]
    except Exception as e:
        print(f"❌ [DB] Error mengambil leads: {e}")
        return []
    finally:
        if conn:
            conn.close()


def extract_lead_from_text(user_message: str) -> dict:
    """
    Mendeteksi dan mengekstrak info kontak/event dari SATU pesan user.
    """
    extracted = {}
    msg_lower = user_message.lower()
    
    # Deteksi No HP (misal: 0812..., 62812..., +62812...)
    phone_match = re.search(r'(\+?62|0)8[1-9][0-9]{7,10}', user_message)
    if phone_match:
        extracted['phone'] = phone_match.group(0)

    # Deteksi Nama ("1. Reynaldi", "nama: Reynaldi", "nama saya Anisa", "dengan Maya", "perkenalkan Budi")
    num_name_match = re.search(r'(?:^|\n)(?:1[\.\)]\s*|nama\s*[:=-]\s*)([A-Za-z][A-Za-z\s]{1,30})', user_message, re.IGNORECASE)
    if num_name_match:
        raw_name = num_name_match.group(1).strip()
        stop_words = ['mau', 'tanya', 'bisa', 'inisiatif', 'booking', 'gedung', 'rumah', 'wedding', 'lamaran', 'oktober', 'september', 'agustus', 'juli', 'juni', 'mei', 'april', 'maret', 'februari', 'januari']
        if not any(sw in raw_name.lower() for sw in stop_words):
            extracted['customer_name'] = raw_name.title()

    if not extracted.get('customer_name'):
        name_match = re.search(r'(?:nama\s*(?:saya|aku)?\s*:?|dengan|perkenalkan)\s+([A-Za-z][A-Za-z\s]{1,30})', user_message, re.IGNORECASE)
        if name_match:
            raw_name = name_match.group(1).strip()
            stop_words = ['mau', 'tanya', 'bisa', 'inisiatif', 'booking', 'gedung', 'rumah', 'wedding', 'lamaran']
            if not any(sw in raw_name.lower() for sw in stop_words):
                extracted['customer_name'] = raw_name.title()

    # Deteksi Jenis Acara
    if any(k in msg_lower for k in ['nikah', 'wedding', 'resepsi', 'akad', 'menikah', 'pernikahan','unduh mantu']):
        extracted['event_type'] = 'Wedding'
    elif any(k in msg_lower for k in ['lamaran', 'engagement', 'tunangan', 'melamar']):
        extracted['event_type'] = 'Engagement'
    elif any(k in msg_lower for k in ['ulang tahun', 'ultah', 'birthday', 'sweet 17']):
        extracted['event_type'] = 'Birthday'

    # Deteksi Tipe Venue (Gedung vs Rumah)
    if any(k in msg_lower for k in ['gedung', 'hotel', 'hall', 'ballroom', 'masjid', 'resto', 'restaurant', 'convention', 'aula']):
        extracted['venue_type'] = 'Gedung'
    elif any(k in msg_lower for k in ['rumah', 'halaman', 'garasi', 'home', 'kediaman', 'outdoor']):
        extracted['venue_type'] = 'Rumah'

    # Deteksi Tanggal
    parsed_date = parse_indonesian_date(user_message)
    if parsed_date:
        extracted['event_date'] = str(parsed_date)

    # Deteksi Lokasi/Kota
    location_keywords = [
        'jakarta', 'bandung', 'surabaya', 'bekasi', 'tangerang', 'depok', 'bogor',
        'semarang', 'yogyakarta', 'jogja', 'malang', 'solo', 'medan', 'makassar',
        'bali', 'denpasar', 'palembang', 'batam', 'pekanbaru', 'manado',
        'cibubur', 'bsd', 'serpong', 'pondok indah', 'kelapa gading', 'pik',
        'kemang', 'cilandak', 'menteng', 'senayan', 'kuningan', 'sudirman',
    ]
    for loc in location_keywords:
        if loc in msg_lower:
            extracted['location'] = loc.title()
            break

    return extracted


def extract_lead_from_conversation(current_message: str, history: list = None, wa_name: str = None, wa_phone: str = None) -> dict:
    """
    Mengakumulasi data lead dari SELURUH percakapan (current message + history).
    HANYA nomor HP (wa_phone) yang diambil dari Meta Cloud API.
    Nama klien (customer_name) WAJIB diekstrak langsung dari isi pesan percakapan user!
    
    Returns dict dengan keys:
        customer_name, phone, event_date, event_type, venue_type, location,
        package (gabungan event_type + venue_type), is_complete (bool)
    """
    accumulated = {}

    # 1. Scan semua pesan user dari history
    if history and isinstance(history, list):
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                partial = extract_lead_from_text(msg["content"])
                # Merge: field baru menimpa yang lama (data terbaru menang)
                for key, val in partial.items():
                    if val:
                        accumulated[key] = val

    # 2. Scan pesan saat ini (prioritas tertinggi)
    current_extracted = extract_lead_from_text(current_message)
    for key, val in current_extracted.items():
        if val:
            accumulated[key] = val

    # 3. HANYA No HP (wa_phone) yang di-override dari Meta Cloud API / WhatsApp webhook
    if wa_phone and not accumulated.get('phone'):
        accumulated['phone'] = wa_phone

    # CATATAN: wa_name DIBUANG / TIDAK DIPAKAI agar nama klien murni dari teks chat!

    # 4. Generate package name (gabungan event_type + venue_type)
    event_type = accumulated.get('event_type')
    venue_type = accumulated.get('venue_type')
    if event_type and venue_type:
        accumulated['package'] = f"{event_type} {venue_type}"

    # 5. Cek kelengkapan 4 filter wajib (Nama, Tanggal, EventType, VenueType)
    accumulated['is_complete'] = all([
        accumulated.get('customer_name'),
        accumulated.get('event_date'),
        accumulated.get('event_type'),
        accumulated.get('venue_type'),
    ])

    return accumulated


MONTH_NAMES = {
    'januari': 1, 'jan': 1,
    'februari': 2, 'feb': 2,
    'maret': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'mei': 5,
    'juni': 6, 'jun': 6,
    'juli': 7, 'jul': 7,
    'agustus': 8, 'agus': 8, 'agu': 8,
    'september': 9, 'sep': 9,
    'oktober': 10, 'okt': 10,
    'november': 11, 'nov': 11,
    'desember': 12, 'des': 12
}


def parse_indonesian_date(text: str):
    """
    Mengubah teks seperti '28 agustus', '15 September 2026', '2026-08-28' menjadi datetime.date
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    # Cek format YYYY-MM-DD
    iso_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', text_lower)
    if iso_match:
        try:
            return datetime(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))).date()
        except ValueError:
            pass

    # Cek format DD/MM/YYYY
    slash_match = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', text_lower)
    if slash_match:
        try:
            return datetime(int(slash_match.group(3)), int(slash_match.group(2)), int(slash_match.group(1))).date()
        except ValueError:
            pass

    # Cek format '28 agustus' atau '28 agustus 2026'
    for month_name, month_num in MONTH_NAMES.items():
        pattern = r'(\d{1,2})\s+' + month_name + r'(\s+(\d{4}))?'
        match = re.search(pattern, text_lower)
        if match:
            day = int(match.group(1))
            year = int(match.group(3)) if match.group(3) else datetime.now().year
            try:
                return datetime(year, month_num, day).date()
            except ValueError:
                pass

    return None


def check_date_availability(event_date) -> dict:
    """
    Memeriksa ketersediaan tanggal acara di database Supabase.
    Tanggal dianggap TERISI jika statusnya 'Booked', 'DpPaid', atau 'Completed'.
    """
    if isinstance(event_date, str):
        parsed_date = parse_indonesian_date(event_date)
    elif isinstance(event_date, datetime):
        parsed_date = event_date.date()
    else:
        parsed_date = event_date

    if not parsed_date:
        return {"status": "invalid_date", "available": True}

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, customer_name, event_type, package, status, location
            FROM leads
            WHERE event_date = %s AND status::text IN ('Booked', 'DpPaid', 'Completed');
        """, (parsed_date,))
        bookings = cursor.fetchall()
        
        date_formatted = parsed_date.strftime("%d %B %Y")
        
        if bookings:
            return {
                "available": False,
                "parsed_date": str(parsed_date),
                "date_formatted": date_formatted,
                "existing_bookings": [dict(b) for b in bookings]
            }
        else:
            return {
                "available": True,
                "parsed_date": str(parsed_date),
                "date_formatted": date_formatted,
                "existing_bookings": []
            }
    except Exception as e:
        print(f"❌ [DB Error check_date_availability]: {e}")
        return {"available": True, "error": str(e)}
    finally:
        if conn:
            conn.close()



if __name__ == "__main__":
    print("🔌 Tes Koneksi Database Supabase...")
    try:
        leads = get_recent_leads(5)
        print(f"✅ Terhubung! Total {len(leads)} lead ditemukan di database:")
        for l in leads:
            print(f"   - {l['customer_name']} | {l['event_type']} | {l['status']}")
    except Exception as err:
        print(f"❌ Koneksi gagal: {err}")
