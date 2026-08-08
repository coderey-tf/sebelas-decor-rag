"""
test_rag.py — Script Testing RAG Engine Sebelas Decor
===================================================
Menjalankan pengujian otomatis beberapa scenario pertanyaan random pelanggan
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag_engine import query_rag

test_queries = [
    {
        "scenario": "1. Minta Pricelist & Katalog",
        "query": "Halo, boleh minta pricelist dekorasinya?",
        "history": []
    },
    {
        "scenario": "2. Menyebutkan Detail Acara (Wedding di Jakarta)",
        "query": "Untuk acara Wedding di Jakarta tanggal 15 Oktober 2026",
        "history": [
            {"role": "user", "content": "Halo, boleh minta pricelist dekorasinya?"},
            {"role": "assistant", "content": "Halo! Berikut link pricelist kami..."}
        ]
    },
    {
        "scenario": "3. Pertanyaan Pembayaran & DP",
        "query": "Berapa minimal DP untuk booking tanggal dan kapan pelunasannya?",
        "history": []
    },
    {
        "scenario": "4. Pertanyaan Ketentuan Revisi",
        "query": "Apakah ada revisi gratis kalau desainnya kurang pas?",
        "history": []
    },
    {
        "scenario": "5. Pertanyaan Custom Tema & Lokasi",
        "query": "Saya mau tema Rustic Minimalist untuk outdoor di Bogor, apakah bisa?",
        "history": []
    },
    {
        "scenario": "6. Pertanyaan Add-On",
        "query": "Bisa tambahkan smoke machine dan neon sign custom?",
        "history": []
    }
]

def run_tests():
    print("=" * 65)
    print("🌸 SEBELAS DECOR — AUTOMATED RAG ENGINE TEST SUITE 🌸")
    print("=" * 65 + "\n")

    for item in test_queries:
        print(f"📌 SCENARIO: {item['scenario']}")
        print(f"👤 KLIEN   : {item['query']}")
        print("-" * 65)
        
        try:
            reply = query_rag(item['query'], history=item['history'])
            print(f"🤖 BOT     :\n{reply}\n")
        except Exception as e:
            print(f"❌ ERROR   : {e}\n")
        
        print("=" * 65 + "\n")

if __name__ == "__main__":
    run_tests()
