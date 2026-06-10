# Klasifikasi Sampah Bertingkat Menggunakan Pendekatan Hierarchical Classification

Proyek ini merupakan implementasi sistem klasifikasi citra sampah secara bertingkat (Hierarchical Classification) untuk mengelompokkan data ke dalam 9 kelas akhir. Pendekatan ini membagi beban kerja evaluasi menjadi dua tingkatan arsitektur guna meminimalkan ruang pencarian kelas dan mengoptimalkan akurasi klasifikasi akhir.

## Arsitektur Sistem

Sistem ini memecah proses klasifikasi konvensional menjadi jalur keputusan bertingkat sebagai berikut:

1. **Level 1 (Root Model):** Menggunakan arsitektur **Vision Transformer (ViT-B/16)** untuk mengidentifikasi kategori makro, yaitu memisahkan citra menjadi kelompok `organic` atau `inorganic`.
2. **Level 2 (Sub-Models):**
   * **Cabang Organik:** Menggunakan arsitektur **MobileNetV3-Large** untuk klasifikasi efisien pada 2 kelas spesifik: `Food Organics` dan `Vegetation`.
   * **Cabang Anorganik:** Menggunakan arsitektur **ConvNeXt-Tiny** untuk mengatasi tantangan variasi fitur yang kompleks pada 7 kelas spesifik: `Cardboard`, `Glass`, `Metal`, `Miscellaneous Trash`, `Paper`, `Plastic`, dan `Textile Trash`.

---

## Struktur Direktori

```text
projek-tubes-trash/
├── data/
│   ├── raw/                 # Dataset asli berstruktur 9 kelas dari sumber data
│   └── processed/           # Dataset hasil otomatisasi pemisahan (Train/Val/Test) per level
├── notebooks/               # Dokumentasi eksperimen terstruktur (.ipynb)
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_train_root_model.ipynb
│   ├── 04_train_sub_models.ipynb
│   └── 05_inference_pipeline.ipynb
├── src/                     # Source code utama berbasis modul python (.py)
│   ├── __init__.py          # Identifikasi python package
│   ├── dataset.py           # Pipeline augmentasi data dan fungsi loader
│   ├── models.py            # Definisi modifikasi arsitektur ViT, MobileNetV3, dan ConvNeXt
│   └── utils.py             # Fungsi utilitas plotting kurva dan matriks evaluasi
├── saved_models/            # Direktori penyimpanan berkas bobot model terbaik (.pth)
├── reports/                 # Hasil visualisasi untuk dokumentasi laporan
│   └── figures/             # Grafik kurva performa, confusion matrix, dan analisis sampel
├── .gitignore               # Konfigurasi pengabaian berkas pelacakan Git
├── README.md                # Dokumentasi utama proyek
└── requirements.txt         # Daftar dependensi library Python
```

## Spesifikasi Kebutuhan Sistem

Seluruh komputasi dioptimalkan menggunakan akselerasi perangkat keras berbasis NVIDIA CUDA. Berikut adalah pustaka utama yang digunakan dalam lingkungan Python:

* PyTorch >= 2.1.0
* Torchvision >= 0.16.0
* NumPy >= 1.24.0
* Pandas >= 2.0.0
* Scikit-Learn >= 1.3.0
* Matplotlib >= 3.7.0
* Seaborn >= 0.12.0
* TQDM >= 4.65.0

## Alur Eksekusi Pipeline

Untuk menjalankan proyek dari tahap awal hingga evaluasi akhir, eksekusi berkas notebook di dalam direktori `notebooks/` secara berurutan:

### 1. Analisis Data Eksploratif

Buka dan jalankan `01_exploratory_data_analysis.ipynb` untuk:

* Melakukan validasi total citra dan mendeteksi adanya ketidakseimbangan kelas (class imbalance).
* Menganalisis statistik dimensi gambar dan aspek rasio.
* Melakukan verifikasi berkas citra untuk memastikan tidak ada data yang rusak (corrupted images).

### 2. Preprocessing dan Stratified Splitting

Buka dan jalankan `02_data_preprocessing.ipynb` untuk:

* Membagi dataset menjadi proporsi data Train (80%), Validation (10%), dan Test (10%).
* Menerapkan teknik Stratified Split guna mempertahankan representasi distribusi kelas asli pada tiap subset data.
* Menyusun ulang struktur folder ke dalam format hierarki di dalam direktori `data/processed/`.

### 3. Pelatihan Model Gerbang Utama (Root Level)

Buka dan jalankan `03_train_root_model.ipynb` untuk:

* Memuat arsitektur ViT-B/16 dengan bobot prapelatihan ImageNet.
* Melakukan proses fine-tuning langsung menggunakan flat-loop untuk memisahkan kategori organik dan anorganik.
* Menyimpan bobot terbaik berdasarkan performa data validasi ke dalam `saved_models/root_model_best.pth`.

### 4. Pelatihan Sub-Model Cabang Spesifik

Buka dan jalankan `04_train_sub_models.ipynb` untuk:

* Melatih MobileNetV3-Large pada subset data organik (2 kelas).
* Melatih ConvNeXt-Tiny pada subset data anorganik (7 kelas).
* Memantau proses perulangan menggunakan visualisasi TQDM dan menyimpan masing-masing berkas bobot terbaik `.pth`.

### 5. Evaluasi Pipeline Akhir (Inference)

Buka dan jalankan `05_inference_pipeline.ipynb` untuk:

* Mengintegrasikan ketiga model terbaik ke dalam satu alur inferensi end-to-end.
* Menguji performa sistem klasifikasi bertingkat menggunakan keseluruhan data uji independen.
* Memproduksi metrik Classification Report komplit beserta visualisasi Confusion Matrix komprehensif untuk 9 kelas akhir.
* Menampilkan sampel evaluasi kualitatif berupa representasi citra dengan tebakan benar (success cases) dan tebakan salah (failure cases).
