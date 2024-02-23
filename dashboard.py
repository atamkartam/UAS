import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Fungsi untuk memuat data dari file CSV dan memformat missing values
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/atamkartam/UAS/main/day.csv")
    df['dteday'] = pd.to_datetime(df['dteday'])  # Mengubah kolom 'dteday' ke format datetime
    missing_value_formats = ['N.A', 'na', 'n.a.', 'n/a', '?', '-']  # Definisikan format missing values
    df.replace(missing_value_formats, np.nan, inplace=True)  # Format missing values dengan NaN
    return df

# Fungsi untuk melakukan analisis awal pada data termasuk missing values, duplikat, dan format data
def analisis_review(df):
    st.header("Analisis Review Data")
    st.subheader("Dataset")
    st.write("Berikut adalah preview dari dataset:")
    st.dataframe(df)  # Menampilkan DataFrame

    # Memeriksa dan menampilkan informasi missing values
    st.subheader("Missing Value atau Nilai Hilang")
    if df.isnull().values.any():
        st.write("Terdapat missing values dalam dataset.")
        missing_info = df.isnull().sum()
        st.write(missing_info[missing_info > 0])
    else:
        st.write("Tidak terdapat missing values dalam dataset.")

    # Memeriksa dan menampilkan informasi duplikat data
    st.subheader("Duplikat Data")
    if df.duplicated().any():
        st.write("Terdapat duplikat data dalam dataset.")
        st.write(df[df.duplicated(keep=False)])
    else:
        st.write("Tidak terdapat duplikat data dalam dataset.")

    # Menampilkan tipe data untuk setiap kolom
    st.subheader("Data Formatting")
    st.write("Tipe data setiap kolom:")
    st.write(df.dtypes)

    st.write("Dari output di atas, terlihat semua fitur sepeti missing value, data duplikat dan data formatting sudah sesuai jika kita lihat datanya di dataframe, maka kita tidak perlu melakukan cleaning data pada dataset kali ini.")

# Fungsi untuk visualisasi data
def visualize_data(df):
    st.header("Hasil Analisis Data")
    # Visualisasi 1: Peminjam Sepeda Terdaftar dan Casual pada Tahun 2012
    st.subheader("Visualisasi Data 1")
    df_2012 = df[(df['dteday'] >= '2012-01-01') & (df['dteday'] <= '2012-12-31')]
    df_monthly = df_2012.groupby(df_2012['dteday'].dt.month)[['registered', 'casual']].sum().reset_index()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(16,8))
    bar_width = 0.4
    plt.bar(df_monthly['dteday'] - bar_width/2, df_monthly['registered'], width=bar_width, label='Registered', alpha=0.8)
    plt.bar(df_monthly['dteday'] + bar_width/2, df_monthly['casual'], width=bar_width, label='Casual', alpha=0.8)
    plt.title('Jumlah Peminjam Sepeda Terdaftar dan Casual pada Tahun 2012 Berdasarkan Bulan', size=18)
    plt.xlabel('\nBulan', size=14)
    plt.ylabel('Total Peminjam\n', size=14)
    plt.xticks(df_monthly['dteday'], months, size=14)
    plt.legend()
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 1"):
        st.write('Pada umumnya, jumlah peminjam sepeda terdaftar (Registered) lebih tinggi daripada peminjam sepeda acak (Casual) sepanjang tahun 2012. Puncak   aktivitas peminjaman terdaftar tampaknya terjadi pada bulan september dengan jumlah yaitu 174795 peminjam dan Puncak aktivitas peminjaman casual tampaknya terjadi pada bulan mei dengan jumlah yaitu 44235 peminjam, yang mungkin disebabkan oleh cuaca yang lebih hangat dan kondisi yang lebih baik untuk bersepeda. Dalam keseluruhan, peminjaman sepeda terdaftar cenderung lebih stabil dan dapat diprediksi, sementara peminjaman casual lebih bervariasi dan mungkin dipengaruhi oleh faktor-faktor seperti liburan atau cuaca yang tidak terduga.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Visualisasi Data 2: Pengaruh Kecepatan Angin Terhadap Penggunaan Sepeda
    st.subheader("Visualisasi Data 2")
    bulan_map = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni',
                7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
    df_2011_weekday = df[(df['yr'] == 0) & (df['workingday'] == 1)]
    avg_wind_speed_by_month = df_2011_weekday.groupby(df_2011_weekday['dteday'].dt.month)['windspeed'].mean()
    avg_wind_speed_by_month.index = avg_wind_speed_by_month.index.map(bulan_map)
    plt.figure(figsize=(10, 8))
    plt.pie(avg_wind_speed_by_month, labels=avg_wind_speed_by_month.index, autopct='%1.1f%%', startangle=140)
    plt.title('Pengaruh Kecepatan Angin Terhadap Penggunaan Sepeda pada Hari Kerja Tahun 2011 (Berdasarkan Bulan)')
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 2"):
        st.write('Dapat disimpulkan bahwa kecepatan angin tidak memiliki pengaruh yang konsisten terhadap penggunaan sepeda. Meskipun terdapat variasi kecepatan angin dari bulan ke bulan, penggunaan sepeda cenderung tetap tinggi pada beberapa bulan meskipun kecepatan angin relatif tinggi. Temuan ini menunjukkan bahwa faktor-faktor lain, seperti cuaca atau suhu, mungkin juga berkontribusi terhadap pola penggunaan sepeda pada hari kerja.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Analisis Pertanyaan 3: Perkembangan Jumlah Penyewa Sepeda Seiring Waktu
    st.subheader("Visualisasi Data 3")
    plt.figure(figsize=(10, 6))
    plt.plot(df['dteday'], df['cnt'], label='Total Penyewa Sepeda')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Sepeda Disewa')
    plt.title('Perkembangan Jumlah Penyewa Sepeda Seiring Waktu')
    plt.legend()
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 3"):
        st.write('Melalui pemantauan perkembangan jumlah penyewa sepeda sepanjang waktu, terlihat bahwa terdapat kecenderungan peningkatan secara umum, meskipun dengan beberapa fluktuasi. Pola ini bisa mengindikasikan adanya tren musiman atau faktor-faktor lain yang berpengaruh pada keberlangsungan pertumbuhan tersebut.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Analisis Pertanyaan 4: Perbedaan Jumlah Penyewa Sepeda antara Hari Kerja dan Hari Libur
    st.subheader("Visualisasi Data 4")
    plt.figure(figsize=(8, 5))
    sns.barplot(x='workingday', y='cnt', data=df, estimator=sum, errorbar=None)
    plt.xlabel('Hari Kerja (1) / Hari Libur (0)')
    plt.ylabel('Jumlah Sepeda Disewa')
    plt.title('Perbedaan Jumlah Penyewa Sepeda antara Hari Kerja dan Hari Libur')
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 4"):
        st.write('Analisis menunjukkan bahwa jumlah penyewa sepeda lebih tinggi pada hari kerja dibandingkan dengan hari libur. Hal ini kemungkinan besar dipengaruhi oleh kebutuhan transportasi sehari-hari pada hari kerja, sedangkan pada hari libur, aktivitas sepeda bisa lebih bersifat rekreasi.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Analisis Pertanyaan 5: Korelasi antara Suhu dan Jumlah Penyewa Sepeda
    st.subheader("Visualisasi Data 5")
    plt.figure(figsize=(8, 5))
    plt.scatter(df['temp'], df['cnt'])
    plt.xlabel('Suhu ')
    plt.ylabel('Jumlah Sepeda Disewa')
    plt.title('Korelasi antara Suhu dan Jumlah Penyewa Sepeda')
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 5"):
        st.write('Korelasi positif antara suhu dan jumlah penyewa sepeda menarik untuk dicatat. Saat suhu meningkat, jumlah penyewa sepeda juga cenderung meningkat. Hal ini menunjukkan bahwa preferensi pengguna sepeda terkait dengan kondisi cuaca yang lebih hangat.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Analisis Pertanyaan 6: Pengaruh Musim terhadap Jumlah Penyewa Sepeda
    st.subheader("Visualisasi Data 6")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='cnt', data=df, estimator=sum, errorbar=None)
    plt.xlabel('Musim')
    plt.ylabel('Jumlah Sepeda Disewa')
    plt.title('Pengaruh Musim terhadap Jumlah Penyewa Sepeda')
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 6"):
        st.write('Terlihat bahwa musim memengaruhi jumlah penyewa sepeda, dengan tingkat penggunaan yang lebih tinggi selama musim panas dan musim semi. Faktor iklim ini dapat menjadi pertimbangan penting dalam merencanakan layanan sepeda dan kebijakan pengembangan.')
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Analisis Pertanyaan 7: Pengaruh Kondisi Cuaca terhadap Jumlah Penyewa Sepeda
    st.subheader("Visualisasi Data 7")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='weathersit', y='cnt', data=df, estimator=sum, errorbar=None)
    plt.xlabel('Kondisi Cuaca')
    plt.ylabel('Jumlah Sepeda Disewa')
    plt.title('Pengaruh Kondisi Cuaca terhadap Jumlah Penyewa Sepeda')
    st.pyplot(plt)
    # Expander Grafik
    with st.expander("Kesimpulan  Visualisasi Data 7"):
        st.write('Cuaca yang cerah atau berawan cenderung meningkatkan jumlah penyewa sepeda, sementara kondisi cuaca buruk dapat berdampak negatif. Kesimpulan ini memberikan gambaran holistik tentang bagaimana faktor-faktor lingkungan dan kegiatan sehari-hari berkontribusi terhadap pola penggunaan sepeda.')

def analisis_lanjutan(df):
    st.header("Teknik Analisis Lanjutan")
    # Bagian 1: KMeans Clustering
    st.subheader("KMeans Clustering")
    X = df[['temp', 'atemp', 'hum', 'windspeed']]
    X_normalized = (X - X.mean()) / X.std()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X_normalized)
    df['cluster'] = kmeans.labels_
    sns.scatterplot(x='temp', y='hum', hue='cluster', data=df)
    plt.title('Clustering Data Cuaca')
    plt.xlabel('Suhu')
    plt.ylabel('Kelembaban')
    st.pyplot(plt)
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    # Bagian 2: Seasonal Decomposition
    st.subheader("Seasonal Decomposition")
    ts_data = df.set_index('dteday')['cnt']
    decomposition = seasonal_decompose(ts_data, model='additive')
    decomposition.plot()
    st.pyplot(plt)
    st.write('<hr>', unsafe_allow_html=True) #hr Garis Pemisah

    
# Bagian 3: ARIMA Prediction
    st.subheader("ARIMA Prediction")
    # Memilih hanya kolom 'dteday' dan 'cnt' untuk analisis
    df_time_series = df.set_index('dteday')['cnt']
    
    # Menggunakan model ARIMA
    model = ARIMA(df_time_series, order=(5,1,2))  # Parameter order (p,d,q) harus disesuaikan berdasarkan data
    model_fit = model.fit()
    
    # Melakukan prediksi untuk 30 hari ke depan
    forecast = model_fit.forecast(steps=30)  # Prediksi untuk 30 hari kedepan
    
    # Visualisasi prediksi
    plt.figure(figsize=(10, 5))
    plt.plot(forecast, label='Forecast')
    plt.title('30-Day Forecast')
    plt.xlabel('Time')
    plt.ylabel('Number of Rentals')
    plt.legend()
    st.pyplot(plt)

def show_team_members():
    st.header("Anggota Kelompok JOSU")
    st.markdown("""
    - Kelompok : JOSU
    - Anggota : <br>
        - 10122352 - Muhammad Insani Imam Utomo<br>
        - 10122353 - Krisnover Aritonang<br>
        - 10122357 - Mohamad Trio Solehudin<br>
        - 10122367 - Atam Kartam<br>
        - 10122376 - A.shifa Muhammad Yusuf<br>
        - 10122378 - Muhammad Raizidan Maliq<br>
    """, unsafe_allow_html=True)

def main():
    # Memuat data dari file CSV menggunakan fungsi load_data()
    df = load_data()

    # Membuat sidebar dengan Streamlit
    with st.sidebar:
        # Menampilkan opsi menu untuk memilih tampilan dashboard
        selected = option_menu("Menu", ["Dashboard", "Anggota Kelompok"], icons=["easel2", "people"], menu_icon="cast", default_index=0)
    
    # Jika opsi yang dipilih adalah "Dashboard", maka akan menampilkan dashboard analisis penyewaan sepeda
    if selected == "Dashboard":
        st.header("Dashboard Analisis Penyewaan Sepeda")

        # Membuat tab untuk "Data Wrangling" dan "Visualisasi Data"
        tab1, tab2 = st.tabs(["Data Wrangling", "Visualisasi Data"])

    # Di dalam tab "Data Wrangling", akan menampilkan hasil analisis data menggunakan fungsi analisis_review()
        with tab1:
            analisis_review(df)

        # Di dalam tab "Visualisasi Data", akan menampilkan hasil visualisasi data menggunakan fungsi visualize_data()
        with tab2:
            visualize_data(df), analisis_lanjutan(df)
    # Jika opsi yang dipilih adalah "Anggota Kelompok", maka akan menampilkan informasi anggota kelompok
    elif selected == "Anggota Kelompok":
        show_team_members()
    
    
    

# Memanggil fungsi main() jika script dijalankan sebagai program utama
if __name__ == "__main__":
    main()