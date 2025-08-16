
# =============================
# Right Portfolio Sidebar (Fixed)
# =============================
PORTFOLIO = {
    "name": "Sri Supatmi",
    "title": "Data Analyst / Aspiring Data Scientist",
    "tagline": "Turning data into actionable strategy with sharp insights that drive real impact—fast, precise, measurable with: Python • SQL • Power BI • Streamlit.",
    "location": "Gunma, Japan",
    "email": "srisupatmi119@gmail.com",
    "phone": "+81 8070 552 365",
    "linkedin": "https://linkedin.com/in/srisupatmi",
    "skills": ["Python", "SQL", "Power BI", "Streamlit", "Excel"],
    "hobbies": ["📚 Artikel data science", "🧘‍♀️ Yoga & mindfulness", "🌏 Traveling & budaya"],
    "photo": "profile.jpeg",
    "cv_path": "cv.pdf"
}

RESUME_DATA = {
    "summary": 'Aspiring Data Scientist with an Accounting associate degree and 4 years in technology and food manufacturing. Skilled in Python, SQL, Power BI, VS Code, Pandas, and NumPy. Experienced in data preprocessing, EDA, statistical/statistical and predictive modeling, and communicating insights to both technical and non-technical stakeholders. Passionate about using data to drive decisions and optimize financial processes, and eager to grow in a challenging data science role',
    "skills": ': Indonesian – Native English – Professional Working Proficiency Japanese – JLPT N4',
    "experience": 'Tatsukichi Co., Ltd. – Japan        2024 - Present Food processing Operator Process food materials to produce high-quality products, ensuring consistent production outcomes.Collaborate with the team to ensure optimal material availability and production quantities.PT. Putra Jaya Nanas – Indonesia          2022Financial Systems Operator (Intern)Managed financial data using company systems to ensure accurate reporting and financial control.Improved financial data accuracy and processing speed by 55%, streamlining workflows through the implementation of automated',
    "education": 'DIBIMBING.ID – Jakarta, Indonesia  2024 - 2025Data Science Program | Score: 90/100Focus on statistics, data analytics, machine learning, and data visualization Technical',
    "certifications": '',
    "projects": '',
    "languages": ': Python, Java, SQLData Analysis & Visualization',
}

def _file_exists(p):
    import os
    try:
        return os.path.exists(p)
    except Exception:
        return False

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# === Fungsi Visualisasi Nyata ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Prediksi Harga Vila Bukit Vista Menggunakan Model Machine Learning XGBoost", layout="wide", initial_sidebar_state="expanded")

def render_profile_card():
    import os
    def _exists(p):
        try: return os.path.exists(p)
        except: return False

    # Card wrapper
    st.markdown("<div class='card sticky'>", unsafe_allow_html=True)

    # Header (gradient)
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #a18cd1, #fbc2eb);
                    color:#fff; padding:12px 14px; border-radius:16px;">
          <div style="font-weight:800; font-size:16px; line-height:1;">{name}</div>
          <div style="opacity:.95; font-size:12px;">{title}</div>
        </div>
        """.format(name=PORTFOLIO.get("name",""), title=PORTFOLIO.get("title","")),
        unsafe_allow_html=True
    )

    # Photo
    if _exists(PORTFOLIO.get("photo","")):
        st.image(PORTFOLIO["photo"], width=180)

    # Contacts
    if PORTFOLIO.get("tagline"):
        st.caption(PORTFOLIO.get("tagline"))
    if PORTFOLIO.get("location"):
        st.write(f"📍 {PORTFOLIO['location']}")
    if PORTFOLIO.get("linkedin"):
        st.write(f"🔗 LinkedIn: {PORTFOLIO['linkedin']}")
    if PORTFOLIO.get("email"):
        st.write(f"✉️ {PORTFOLIO['email']}")
    if PORTFOLIO.get("phone"):
        st.write(f"📱 {PORTFOLIO['phone']}")

    # Skills
    if PORTFOLIO.get("skills"):
        badges = " ".join([f"<span style='display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;padding:4px 8px;border-radius:999px;font-size:12px;margin:2px;'>{s}</span>" for s in PORTFOLIO["skills"]])
        st.markdown("**Skills**", unsafe_allow_html=True)
        st.markdown(badges, unsafe_allow_html=True)

    # Resume expander
    with st.expander("Resume", expanded=False):
        if RESUME_DATA.get("summary"):
            st.markdown("**Ringkasan**")
            st.write(RESUME_DATA["summary"])
        if RESUME_DATA.get("experience"):
            st.markdown("**Experience**")
            st.write(RESUME_DATA["experience"][:1200] + ("..." if len(RESUME_DATA["experience"])>1200 else ""))
        if RESUME_DATA.get("education"):
            st.markdown("**Education**")
            st.write(RESUME_DATA["education"])
        if RESUME_DATA.get("certifications"):
            st.markdown("**Certifications**")
            st.write(RESUME_DATA["certifications"])
        if RESUME_DATA.get("projects"):
            st.markdown("**Projects**")
            st.write(RESUME_DATA["projects"][:800] + ("..." if len(RESUME_DATA["projects"])>800 else ""))
        if RESUME_DATA.get("languages"):
            st.markdown("**Languages**")
            st.write(RESUME_DATA["languages"])

    # CV button
    cvp = PORTFOLIO.get("cv_path","")
    if _exists(cvp):
        with open(cvp, "rb") as f:
            st.download_button("📥 Unduh CV", f, file_name="Sri_Supatmi_CV.pdf", mime="application/pdf")

    st.markdown("</div>", unsafe_allow_html=True)


# =====================
# Global UI Enhancements
# =====================
def _inject_global_ui():
    st.markdown(
        """
        <style>
        :root{
          --card-bg:#ffffff; --card-border:#e5e7eb; --shadow:0 10px 24px rgba(0,0,0,.05);
          --muted:#64748b; --accent:#4f46e5;
        }
        .main .block-container{ max-width:1200px; padding-top:22px; }
        h1,h2,h3{ letter-spacing:.2px; }
        .section-title{
           font-weight:800; font-size:1.15rem; margin:10px 0 8px;
           padding-bottom:6px; border-bottom:1px dashed var(--card-border);
        }
        .card{
           background:var(--card-bg); border:1px solid var(--card-border);
           border-radius:18px; padding:14px; box-shadow:var(--shadow); margin:8px 0 18px;
        }
        .caption{ color:var(--muted); font-size:.9rem; margin-top:4px;}
        .sticky{position:sticky;top:16px}
.profile-photo img{border-radius:14px}
</style>
        """,
        unsafe_allow_html=True)

def card_plot(fig):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

def card_plotly(fig):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

_inject_global_ui = _inject_global_ui
_inject_global_ui()

# ----------------------
# Data & Model Loaders
# ----------------------
@st.cache_data
def load_data():
    # Sumber utama dari notebook: combined_dataframe.csv
    return pd.read_csv("combined_dataframe.csv")

@st.cache_resource
def load_model():
    # Jika model tidak tersedia, tampilkan info singkat
    try:
        return joblib.load("model_villa.pkl")
    except Exception:
        return None

data = load_data()
model = load_model()

# ----------------------
# Helper Visualisations
# ----------------------
def fig_histogram_outliers():
    numerical_cols = data.select_dtypes(include='number').columns.tolist()
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=data[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    return fig

def fig_scatter_price_vs_bedroom():
    df = data.dropna(subset=['extracted_bedroom_count', 'price_per_night'])
    df = df[df['price_per_night'] > 0]

    ordered_bedrooms = sorted(df['extracted_bedroom_count'].unique())
    df['bedroom_cat'] = pd.Categorical(df['extracted_bedroom_count'], categories=ordered_bedrooms, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['bedroom_cat'].astype(str), df['price_per_night'], alpha=0.6)
    ax.set_xlabel("Jumlah Kamar Tidur")
    ax.set_ylabel("Harga per Malam (SGD)")
    ax.set_title("Sebaran Harga per Malam vs Jumlah Kamar Tidur")
    ax.set_xticks([str(x) for x in ordered_bedrooms])
    fig.tight_layout()
    return fig

def fig_avg_price_bedroom(color='skyblue', title_suffix=""):
    df_group = data.groupby('extracted_bedroom_count')['price_per_night'].mean().reset_index()
    df_group = df_group[df_group['price_per_night'] > 0]
    df_group = df_group.sort_values(by='extracted_bedroom_count')
    df_group['extracted_bedroom_count'] = df_group['extracted_bedroom_count'].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_group['extracted_bedroom_count'], df_group['price_per_night'], color=color, alpha=0.7)

    ax.set_xlabel("Jumlah Kamar Tidur / Number of Bedrooms", fontsize=11)
    ax.set_ylabel("Rata-rata Harga per Malam / Average Price per Night", fontsize=11)
    ax.set_title("Rata-rata Harga per Malam berdasarkan Jumlah Kamar Tidur / Average Price per Night by Number of Bedrooms", fontsize=13)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig

def fig_avg_price_bathroom():
    df_group = data.groupby('extracted_bathroom_count')['price_per_night'].mean().reset_index()
    df_group = df_group[df_group['price_per_night'] > 0]
    df_group = df_group.sort_values(by='extracted_bathroom_count')
    df_group['extracted_bathroom_count'] = df_group['extracted_bathroom_count'].astype(str)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df_group['extracted_bathroom_count'], df_group['price_per_night'], color='salmon')
    ax.set_xlabel("Jumlah Kamar Mandi")
    ax.set_ylabel("Rata-rata Harga (SGD)")
    ax.set_title("Rata-rata Harga per Malam berdasarkan Jumlah Kamar Mandi")
    fig.tight_layout()
    return fig




def fig_violin_price_by_rating():
    # Violin plot dituning agar makin mirip contoh referensi terakhir
    df = data[['rating', 'price_per_night']].dropna()
    df = df[df['price_per_night'] > 0]
    df['rating_round'] = df['rating'].round(1)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    order_vals = sorted(df['rating_round'].unique())
    palette = sns.color_palette("viridis", n_colors=len(order_vals))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x='rating_round',
        y='price_per_night',
        order=order_vals,
        inner='box',
        cut=0,
        bw=0.6,           # sedikit lebih detail (tidak terlalu halus)
        scale='area',     # proporsi area seperti referensi
        width=0.80,       # lebar mirip contoh
        linewidth=1.1,    # outline dan garis box sedikit tebal
        saturation=1.0,
        palette=palette,
        ax=ax)

    # Style dan label meniru referensi
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Harga per Malam (SGD) / Price per Night (SGD)", fontsize=12)
    ax.set_title("Persebaran Harga per Malam berdasarkan Rating / Distribution of Price per Night by Rating", fontsize=13)
    ax.set_ylim(0, 800)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Bingkai lebih jelas seperti contoh
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    return fig

def fig_avg_price_by_location_matplotlib():
    # Ubah one-hot 'location_*' menjadi satu kolom lokasi
    loc_cols = [c for c in data.columns if c.startswith('location_')]
    if len(loc_cols) == 0:
        return None
    df = data.copy()
    df['lokasi'] = df[loc_cols].idxmax(axis=1).str.replace('location_', '')
    grp = (
        df[df['price_per_night'] > 0]
        .groupby('lokasi')['price_per_night']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(grp['lokasi'], grp['price_per_night'], color=plt.cm.viridis(np.linspace(0.15, 0.85, len(grp))))
    ax.set_xlabel("Lokasi / Location")
    ax.set_ylabel("Rata-rata Harga per Malam (SGD) / Average Price per Night (SGD)")
    ax.set_title("Rata-rata Harga per Malam berdasarkan Lokasi / Average Price per Night by Location")
    ax.set_ylim(0, max(grp['price_per_night'])*1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    for bar, val in zip(bars, grp['price_per_night']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+3, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig

# ----------------------
# Header & Sidebar
# ----------------------
st.title("Prediksi Harga Vila Bukit Vista Menggunakan Model Machine Learning XGBoost")
col_left, col_right = st.columns([0.65, 0.35], gap='large')
with col_left:
    st.image("bukit-vista-header-768x432.jpeg", use_column_width=True)
    st.write("""
###  Tentang Proyek Ini
**Airbnb** adalah platform global yang sejak tahun 2008 telah mempertemukan tamu dan tuan rumah dari seluruh dunia.  
Di Indonesia, salah satu mitra utamanya adalah **Bukit Vista** — sebuah perusahaan manajemen properti yang mengelola beragam vila eksotis di Bali, seperti di **Uluwatu**, **Canggu**, **Bingin**, dan kawasan wisata lainnya.
 Setiap vila memiliki keunikan tersendiri dari sisi:

**Lokasi**           
**Desain**             
**Fasilitas**
             
Karena keragaman ini, menentukan harga sewa yang tepat sangatlah penting untuk menjaga daya saing dan meningkatkan keuntungan. Maka dari itu, pendekatan berbasis data dan machine learning menjadi kunci. 
            
Aplikasi ini dikembangkan untuk:
Menganalisis data listing vila Bukit Vista dari platform Airbnb — termasuk fitur seperti harga, lokasi, rating, kapasitas, dan lainnya.
Memprediksi harga sewa per malam secara akurat menggunakan model machine learning seperti:
             
             
**XGBoost**
             
Memberikan rekomendasi harga yang kompetitif berdasarkan hasil analisis data.
Membantu tim manajemen Bukit Vista dalam meningkatkan pendapatan dan menciptakan pengalaman terbaik bagi para tamu.
Selamat menjelajah!""")
with col_right:
    render_profile_card()




lokasi_list = [col for col in data.columns if col.startswith("location_")]

with st.sidebar:
    st.image("logo_bukit_vista.png", use_column_width=True)
    st.markdown("## Room Category")

    lokasi_terpilih = st.selectbox("Pilih Lokasi", [l.replace("location_", "") for l in lokasi_list] if lokasi_list else ["-"])
    rating = st.number_input("Rating", min_value=4.0, max_value=5.0, value=4.8, step=0.05)
    bedroom = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=2)
    bathroom = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2)
    malam = st.number_input("Jumlah Malam", min_value=1, max_value=30, value=1)
    room_count = st.number_input("Total Ruangan", min_value=1, max_value=15, value=3)
    interaction = st.number_input("Bedroom × Rating", min_value=4.0, max_value=50.0, value=9.6, step=0.1)

    input_data = {
        'rating': rating,
        'extracted_bedroom_count': bedroom,
        'extracted_bathroom_count': bathroom,
        'jumlah_malam': malam,
        'room_count': room_count,
        'bedroom_rating_interaction': interaction
    }
    for loc in lokasi_list:
        input_data[loc] = 1.0 if loc == f"location_{lokasi_terpilih}" else 0.0

    input_df = pd.DataFrame([input_data])
    if model is not None:
        prediksi = model.predict(input_df)[0]
        st.markdown("### Hasil Prediksi Harga")
        st.success(f"Estimasi harga per malam: SGD ${prediksi:.2f}")
    else:
        st.info("Model belum tersedia di folder ini (model_villa.pkl).")

# ----------------------
# Visuals (Existing)
# ----------------------
st.subheader("Harga Rata-rata Vila per Lokasi")
data_lokasi = data.copy()
if lokasi_list:
    data_lokasi["lokasi"] = data_lokasi[lokasi_list].idxmax(axis=1).str.replace("location_", "")
    rata2 = data_lokasi.groupby("lokasi")["price_per_night"].mean().reset_index().sort_values(by="price_per_night")
    fig_plotly = px.bar(rata2, x="price_per_night", y="lokasi", orientation="h", title="Harga Rata-rata per Lokasi")
    card_plotly(fig_plotly)

# Tampilkan data mentah
with st.expander("Lihat Data Mentah"):
    st.dataframe(data)


# ----------------------
# NEW REQUIRED VISUALS (placed *below* the salmon bar chart)
# ----------------------
st.markdown("----")
st.markdown("<div class='section-title'>Tambahan Visualisasi</div>", unsafe_allow_html=True)

# A) Violin Plot: Price by Rating (sesuai contoh Gambar 1)
fig_violin = fig_violin_price_by_rating()
card_plot(fig_violin)

# B) Bar Plot: Average Price by Location (sesuai contoh Gambar 2) — versi Matplotlib dengan label nilai
fig_loc = fig_avg_price_by_location_matplotlib()
if fig_loc is not None:
    card_plot(fig_loc)
else:
    st.warning("Kolom lokasi tidak ditemukan (format 'location_*').")


# --- Insight Pie Chart (safe) ---
st.markdown("### Jumlah Malam Menginap")
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(
        """
        **Pasar sangat didominasi short stay, fokus pada kecepatan operasional dan kemudahan check-in/out. Naikkan porsi 2 malam lewat promo Stay 2 Save (5–10%) atau bundling (late check-out, breakfast-to-go)**

        💡 **Aksi Cepat:** Implementasikan self check-in + SOP housekeeping cepat. Aktifkan add-on: late check-out berbayar, airport transfer. Uji A/B promosi 2 malam selama 2–4 minggu; pantau CTR, booking rate, ADR, dan revenue.
        """
    )

with col2:
    try:
        fig, ax = plt.subplots()
        if "jumlah_malam" in data.columns and data["jumlah_malam"].notna().any():
            counts = (
                data["jumlah_malam"]
                .astype(str)
                .value_counts()
                .sort_index()
            )
            total = counts.values.sum()
            if total > 0:
                ax.pie(
                    counts.values,
                    labels=list(counts.index),
                    autopct="%1.1f%%",
                    startangle=90,
                )
            else:
                ax.pie([81.2, 18.8], labels=["1.0", "2.0"], autopct="%1.1f%%", startangle=90)
        else:
            ax.pie([81.2, 18.8], labels=["1.0", "2.0"], autopct="%1.1f%%", startangle=90)

        ax.axis("equal")
        ax.set_title("Distribusi Jumlah Malam (Pie Chart)", fontsize=10)
        st.pyplot(fig)
    except Exception:
        st.info("Pie chart sementara disembunyikan karena ada error kecil pada environment.")

# --- Kanan-kiri block examples ---
st.markdown('')

col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
        <div style='text-align: justify; font-size:16px;'>
        <strong>Harga sewa cenderung meningkat seiring bertambahnya jumlah kamar tidur.</strong><br><br>
        💡 <strong>Peluang:</strong> Strategi awal adalah melakukan evaluasi berkala pada properti 1–2 kamar tidur yang berharga tinggi untuk memastikan “premium”-nya memang didukung fasilitas eksklusif atau lokasi strategis. Sementara itu, identifikasi properti dengan banyak kamar tetapi harga relatif rendah sebagai kandidat optimasi harga. Untuk pemasaran, arahkan unit berkamar sedikit ke pasangan, solo traveler, dan pebisnis; sedangkan unit berkamar banyak difokuskan ke keluarga besar dan grup wisata.
        </div>
        """, unsafe_allow_html=True)
with col2:
    card_plot(fig_scatter_price_vs_bedroom())


col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
        <div style='text-align: justify; font-size:16px;'>
        <strong>
1 kamar tidur → Rata-rata harga terendah (~105 SGD).
2–4 kamar tidur → Harga meningkat bertahap (210 → 250 → 270 SGD).
5 kamar tidur → Mengalami penurunan (~165 SGD) dibanding 4 kamar.
6 kamar tidur → Rata-rata harga tertinggi (~340 SGD).
10 kamar tidur → Sedikit menurun (~300 SGD) dibanding 6 kamar, namun masih tinggi..</strong><br><br>
        💡 <strong>Peluang:</strong> Secara umum, semakin banyak kamar tidur semakin tinggi harga rata-rata, tetapi kategori 5 kamar merupakan karena harga belum optimal atau segmen pasar berbeda—sedangkan 6 dan 10 kamar adalah segmen premium (keluarga besar/rombongan), sehingga strategi harga perlu disesuaikan berdasarkan tren jumlah kamar dan segmen pelanggan.
        </div>
        """, unsafe_allow_html=True)
with col2:
    card_plot(fig_avg_price_bedroom(color='cornflowerblue'))


col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("""
        <div style='text-align: justify; font-size:16px;'>
        <strong>Kenaikan harga tidak mulus; ada “loncatan” besar di 6 kamar mandi (segmen grup besar/event).
Artinya, pasar bereaksi kuat pada ambang tertentu, bukan pada setiap penambahan 1 kamar.</strong><br><br>
        💡 <strong>Strategi:</strong> Saran singkat: jalankan A/B test kenaikan harga ±10–15% untuk kategori 5 selama 2–4 minggu sambil memantau CTR, booking rate, dan revenue; untuk segmen 6–10 kamar mandi, tawarkan paket premium (min. 2 malam + add-on event).
Di dashboard tampilkan metrik “harga per kamar mandi”, sertakan interaksi fitur (bath×location/bedroom/rating) dan evaluasi dengan PDP/SHAP, serta gunakan error bar/CI pada chart untuk transparansi variasi dan ukuran sampel.
        """, unsafe_allow_html=True)
with col2:
    card_plot(fig_avg_price_bathroom())

        
