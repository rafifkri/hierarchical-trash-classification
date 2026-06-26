# **Klasifikasi Sampah Bertingkat Menggunakan Pendekatan Hierarchical dengan Vision Transformer pada Level Super-Class dan MobileNetV3 serta ConvNeXt-Tiny pada Sub-Class**

## **Latar Belakang**

Pengelolaan sampah merupakan salah satu tantangan utama dalam menjaga kelestarian lingkungan. Seiring dengan meningkatnya jumlah penduduk dan aktivitas manusia, volume sampah yang dihasilkan terus bertambah sehingga proses pemilahan sampah secara manual menjadi kurang efisien, membutuhkan waktu yang lama, serta rentan terhadap kesalahan manusia (human error). Kondisi tersebut menyebabkan proses daur ulang menjadi kurang optimal karena banyak sampah yang tidak dipisahkan sesuai dengan jenisnya.

Perkembangan teknologi Artificial Intelligence (AI), khususnya Deep Learning, memberikan solusi dalam proses klasifikasi citra secara otomatis. Berbagai penelitian menunjukkan bahwa model Deep Learning mampu mengenali objek berdasarkan karakteristik visual seperti bentuk, warna, tekstur, dan pola dengan tingkat akurasi yang tinggi. Oleh karena itu, teknologi ini dapat dimanfaatkan untuk membantu proses klasifikasi sampah secara otomatis sehingga mendukung sistem pengelolaan sampah yang lebih cepat, akurat, dan efisien.

Pada penelitian ini dikembangkan sistem klasifikasi sampah menggunakan pendekatan Hierarchical Classification. Pendekatan ini membagi proses klasifikasi menjadi dua tahap. Tahap pertama menggunakan Vision Transformer (ViT-B/16) untuk mengklasifikasikan sampah menjadi dua kategori utama, yaitu Organik dan Anorganik. Selanjutnya, hasil klasifikasi tersebut diteruskan ke model khusus, yaitu MobileNetV3 untuk klasifikasi sampah organik dan ConvNeXt-Tiny untuk klasifikasi sampah anorganik. Pendekatan bertingkat ini diharapkan mampu meningkatkan akurasi klasifikasi karena setiap model hanya berfokus pada kategori yang lebih spesifik.

# ♻️ Sistem Klasifikasi Citra Sampah 9 Kelas menggunakan Pipeline Hierarchical Classification Berbasis Arsitektur Heterogen

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Architecture-Heterogeneous_Hybrid-success)](https://github.com/)

Repositori ini berisi implementasi proyek *Deep Learning* untuk klasifikasi citra sampah ke dalam 9 kelas spesifik menggunakan metode **Hierarchical Classification Pipeline** (Klasifikasi Bertingkat). Sistem ini mengombinasikan arsitektur Vision Transformer (ViT) dan Convolutional Neural Network (CNN) modern secara sekuensial guna meminimalkan kerancuan visual antar-kelas.

---

## **Core Novelty & Research Gap Solutions**

Mayoritas model klasifikasi sampah konvensional mengadopsi pendekatan satu tingkat (*flat classification*), di mana satu jaringan saraf dipaksa untuk langsung menebak banyak kelas sekaligus secara acak, seperti yang dilakukan pada riset **Insel dkk. (2026)**, **Aulia dkk. (2024)**, serta **Keskin dkk. (2023)**. Pendekatan ini rentan mengalami penurunan akurasi akibat tingginya ambiguitas visual antar-kategori (*inter-class similarity*), misalnya kertas putih (`Paper`) yang sering tertukar dengan tisu makan bekas (`Food Organics`). 

Meskipun model hibrida mutakhir seperti **HR-ViT (Husein dkk., 2025)** dan **SwinConvNeXt (Madhavi dkk., 2025)** mencoba menggabungkan CNN dan Transformer untuk mendongkrak akurasi, model tunggal tersebut tetap memiliki ruang pencarian (*search space*) yang terlalu luas.

Untuk mengatasi celah riset (*research gap*) tersebut, proyek ini menawarkan **Natural Heterogeneous Hierarchical Classification Pipeline**. Sistem ini menyelaraskan pemilihan model berdasarkan karakteristik alami data dan tingkat kesulitan visual di setiap tingkatan hierarki:

### **Spesifikasi Arsitektur Bertingkat**

| Tingkatan Hierarki | Arsitektur Model | Peran & Mekanisme Spesifik | Dukungan Ilmiah & Jurnal |
| :--- | :--- | :--- | :--- |
| **Level 1: Root Model** | **Vision Transformer (ViT-B/16)** | **Natural Macro Splitting**:<br>Memisahkan citra input ke dalam domain makro: `Organic` vs `Inorganic`. | Memanfaatkan **Mekanisme Self-Attention** untuk menangkap konteks global citra, mengamankan rute awal, dan mencegah efek domino salah klasifikasi sesuai prinsip **Panaroma & Al Rivan (2026)**. |
| **Level 2: Sub-Model A** | **MobileNetV3-Large** | **Lightweight Homogeneous Expert**:<br>Mengklasifikasikan klaster organik menjadi kelas akhir: `Food Organics` dan `Vegetation`. | Klaster organik memiliki jumlah kelas yang sedikit dan fitur yang homogen. Sesuai temuan **Tian dkk. (2024)**, arsitektur ini memaksimalkan kecepatan inferensi dan menghemat memori komputasi. |
| **Level 2: Sub-Model B** | **ConvNeXt-Tiny** | **State-of-the-Art Spatial Expert**:<br>Mengurai 7 kelas anorganik kompleks (`Cardboard`, `Plastic`, `Metal`, dll). | Sampah anorganik di lapangan memiliki distorsi geometri yang tinggi (botol remuk, kaleng penyok). ConvNeXt sangat superior dalam mengekstrak fitur spasial lokal yang rumit berdasarkan keberhasilan riset **TrashNeXt (Tanvir dkk., 2025)**. |

###  **Kontribusi Teknis Utama**
* **Reduksi Ruang Pencarian (*Search Space Reduction*):** Memecah masalah *flat* 9 kelas menjadi keputusan mikro sekuensial ($1 \rightarrow 2$ kelas dan $1 \rightarrow 7$ kelas) untuk mempertajam batas keputusan (*decision boundary*).
* **Penyelarasan Industri (*Natural Splitting*):** Alur kerja Level 1 langsung mencerminkan kebutuhan industri daur ulang nyata, di mana sampah basah (organik) harus dipisahkan sejak awal dari komoditas kering (anorganik) untuk menghindari kontaminasi material.
* **Imbalance-Aware Training via Weighted Loss:** Mengatasi masalah ketimpangan jumlah data lintas kelas (*class imbalance*) yang disoroti oleh **Insel (2026)** menggunakan teknik *Inverse Class Frequency Weighting* secara independen di setiap sub-pipeline demi mengamankan nilai **F1-Macro Score** yang objektif.

---

## **Alur Kerja Sistem (Pipeline)**

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
