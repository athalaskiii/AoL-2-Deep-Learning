# Analisis Sentimen Berbasis Aspek pada Ulasan Hotel Berbahasa Indonesia

Proyek akhir mata kuliah Deep Learning (COMP8044041). Model menentukan sentimen untuk sepuluh aspek pada setiap ulasan hotel dari dataset HoASA (IndoNLU), yaitu AC, air panas, bau, kesan umum, kebersihan, linen, pelayanan, sarapan, TV, dan wifi.

Model akhir adalah ensemble IndoBERT-large dan XLM-R-large dengan AspAttn-Desc, yaitu attention per aspek yang query-nya diinisialisasi dari deskripsi aspek. Macro-F1 pada data test mencapai 0,9498, sedikit di atas hasil terbaik pada paper IndoNLU (0,9427).

## Isi

- `notebooks/`: EDA dan preprocessing, perbandingan model, serta evaluasi dan model akhir
- `src/hoasa/`: kode data, preprocessing, model, metrik, dan pelatihan
- `scripts/`: skrip pelatihan dan eksperimen
- `data/`: data mentah dan hasil preprocessing
- `outputs/`: skor dan probabilitas prediksi setiap run yang dipakai notebook

## Menjalankan

```bash
uv sync
```

Output notebook sudah tersimpan, jadi notebook cukup dibuka. Menjalankan ulang notebook tidak melatih model karena semua angka dihitung dari `outputs/`. Bobot model untuk demo di notebook 3 tidak disertakan karena ukurannya sekitar 3,6 GB. Pelatihan ulang memakai `scripts/train.py` dan memerlukan GPU.
