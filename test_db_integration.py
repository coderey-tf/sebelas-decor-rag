"""
test_db_integration.py — Integration Test RAG Chatbot & Supabase Database
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from db_client import save_lead, get_recent_leads
from app import app

def run_tests():
    print("=" * 60)
    print("🧪 INTEGRATION TEST: RAG Chatbot & Supabase Database")
    print("=" * 60)

    # 1. Test Direct Save Lead
    print("\n1️⃣ Testing save_lead() function...")
    result = save_lead(
        customer_name="Test Lead Integration",
        phone="089999888777",
        event_type="Wedding",
        location="Surabaya Convention Center",
        package="Gold Package",
        theme="Modern Minimalist",
        notes="Test otomatis dari RAG Integration Test"
    )
    assert "error" not in result, f"Save lead failed: {result}"
    print(f"   ✅ Saved successfully! Lead ID: {result.get('id')}")

    # 2. Test Get Recent Leads
    print("\n2️⃣ Testing get_recent_leads()...")
    leads = get_recent_leads(limit=5)
    print(f"   ✅ Total leads retrieved from Supabase: {len(leads)}")
    for l in leads[:3]:
        print(f"      - {l['customer_name']} | {l['phone']} | {l['event_type']}")

    # 3. Test Flask Endpoint GET /api/leads
    print("\n3️⃣ Testing Flask GET /api/leads Endpoint...")
    client = app.test_client()
    res = client.get('/api/leads')
    assert res.status_code == 200
    data = res.get_json()
    print(f"   ✅ Endpoint status 200 OK! Total leads in API response: {data.get('count')}")

    # 4. Test Flask Endpoint POST /api/chat with auto lead capture
    print("\n4️⃣ Testing Flask POST /api/chat with Auto Lead Capture...")
    chat_res = client.post('/api/chat', json={
        "message": "Halo, nama saya Budi Cahyono no HP 081234567891 mau tanya dekorasi pernikahan di Malang"
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.get_json()
    print(f"   ✅ Chat Response received!")
    print(f"   Lead Auto-Saved: {chat_data.get('leadSaved')}")
    print(f"   Bot Reply: {chat_data.get('reply')[:120]}...")

    print("\n" + "=" * 60)
    print("🎉 ALL INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
