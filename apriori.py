import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import association_rules, apriori
import calendar


st.title("Market Basket Analysis Invoice Sayur Menggunakan Algoritma Apriori")
# Menu Navigasi
selected = option_menu(
    menu_title=None, 
    options=["Home", "Visualize", "Rules", "Description"],
    icons=["house", "eye", "list", "file-text"],
    menu_icon="cast",
    default_index = 0,
    orientation="horizontal",

)
# Baca Dataset
df = pd.read_csv('DataSets.csv')
# Mengubah format tanggal menjadi datetime
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
# Membuat kolom bulan dan hari
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday
df['hour'] = df['date_time'].dt.hour
# Mengubah angka menjadi nama bulan dan nama hari
df['month'].replace([i for i in range(1, 12 + 1)],
                   ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                   inplace=True)
df['day'].replace([i for i in range(6 + 1)],
                 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                 inplace=True)

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data['month'].str.contains(month.title())) &
        (data['day'].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else 'No Result'
def hot_encode(x):
    return 1 if x >= 1 else 0

def user_input_features():
    item = st.selectbox('Item', df['Item'].unique())
    month = st.select_slider('Month', df['month'].unique())
    day = st.select_slider('Day', df['day'].unique(), value="Saturday")
    return item, month, day
def parse_list(x):
    x = list(x)
    return x[0] if len(x) == 1 else ", ".join(x)

def return_item_df(items_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)
    
    filtered_data = data.loc[data["antecedents"] == items_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return None
def apriori_rule():
    """
    Applies the Apriori algorithm to find frequent itemsets and association rules.

    Returns:
        pandas.DataFrame: The top 10 association rules sorted by support in descending order.
        The DataFrame contains the following columns: antecedents, consequents, support, confidence, lift.
    """
    item_count = df.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index="Transaction", columns="Item", values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.astype("int32")
    item_count_pivot = item_count_pivot.applymap(hot_encode)

    support = 0.01
    frequent_itemsets = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[
        ["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('support', ascending=False, inplace=True)
    return rules.head(10)


# Halaman Home
if selected == "Home":

    item, month, day = user_input_features()
    data = get_data(month, day)

    if type(data) != type("No Result"):
        item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
        item_count_pivot = item_count.pivot_table(index="Transaction", columns="Item", values='Count', aggfunc='sum').fillna(0)
        item_count_pivot = item_count_pivot.astype("int32")
        item_count_pivot = item_count_pivot.applymap(hot_encode)

        support = 0.01
        frequent_itemsets = apriori(item_count_pivot, min_support=support, use_colnames=True)

        metric = "lift"
        min_threshold = 1
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)[
            ["antecedents", "consequents", "support", "confidence", "lift"]]
        rules.sort_values('confidence', ascending=False, inplace=True)

    if type(data) == type("No Result"):
        st.error(data)

    if type(data) != type("No Result"):
        st.markdown("Hasil Rekomendasi :")
        recommendation_result = return_item_df(item)
    
        if recommendation_result is not None:
            st.success(f"Jika Toko membeli **{item}**, maka Toko juga akan membeli **{recommendation_result[1]}** secara bersamaan")
        else:
            st.warning("Tidak ada rekomendasi yang ditemukan untuk item yang dipilih.")

st.sidebar.header("About", anchor="center")
st.sidebar.markdown("This is a web app to show the implementation of Market Basket Analysis using Apriori Algorithm")

st.sidebar.header("Contact", anchor="center")
st.sidebar.markdown("If you have any questions, feel free to contact me:")
st.sidebar.markdown("Email : mario.aprilnino27@gmail.com")
st.sidebar.markdown("Personal Website: https://marioapn3.github.io/")


st.sidebar.header("Source Code", anchor="center")
st.sidebar.markdown("You can see the source code at:")
st.sidebar.markdown("Github: https://github.com/marioapn3/TugasAkhirDataMining---Apriori-MBA- ")

if selected == "Visualize":
    data = df.copy()

    # 10 ITEM TERBANYAK
    st.subheader("Data 10 Item Paling Banyak Permintaan dari Toko")
    st.write('Dataframe yang digunakan untuk visualisasi adalah 10 item yang paling banyak di request / diminta oleh setiap toko Transmart kepada VeegeFresh yang ada di Indonesia')

    # Buat plot
    plt.figure(figsize=(13, 5))
    sns.set_palette('muted')
    sns.barplot(x=data["Item"].value_counts()[:10].index,
            y=data["Item"].value_counts()[:10].values)
    plt.xlabel("Nama Item")
    plt.ylabel("Jumlah Item")
    plt.xticks(size=13, rotation=45)
    plt.title("10 Item paling laris")

    # Tampilkan plot di Streamlit
    st.pyplot(plt)
    # END 10 ITEM TERBANYAK



    # Visualisasi Jumlah Permintaan Barang per Toko tiap Bulan
    st.subheader("Visualisasi Jumlah Permintaan Barang per Toko tiap Bulan")
    st.write('Dataframe yang digunakan untuk visualisasi adalah jumlah permintaan barang per toko tiap bulan yang request / diminta oleh setiap toko Transmart kepada VeegeFresh yang ada di Indonesia')
 

    data_perbulan = data.groupby('month')['Transaction'].count()
    # Convert month names to indices
    month_indices = [list(calendar.month_name).index(month) for month in data_perbulan.index]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=[calendar.month_name[i] for i in month_indices],
        y=data_perbulan.values,
                color="#FF0000"
    )
    plt.xticks(rotation=-30, size=12)
    plt.title("Jumlah transaksi per toko tiap bulan")
    plt.xlabel("Bulan")
    plt.ylabel("Jumlah Transaksi")

    st.pyplot(plt)
    # End Visualisasi Jumlah Permintaan Barang per Toko tiap Bulan


    # Visualisasi Jumlah Permintaan Barang per Toko tiap Hari
    st.subheader("Visualisasi Jumlah Permintaan Barang per Toko tiap Hari")
    st.write('Dataframe yang digunakan untuk visualisasi adalah jumlah permintaan barang per toko tiap hari yang request / diminta oleh setiap toko Transmart kepada VeegeFresh yang ada di Indonesia')

    # Convert 'day' column to numerical format
    df['day'] = pd.Categorical(df['day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
    df['day_code'] = df['day'].cat.codes

    data_perday = df.groupby('day_code')['Transaction'].count().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=[calendar.day_name[i] for i in data_perday['day_code']],
        y=data_perday['Transaction'].values,
        color="#FF0000"
    )
    plt.xticks(rotation=45)
    plt.title("Jumlah transaksi Per Hari")
    plt.xlabel("Hari")
    plt.ylabel("Jumlah Transaksi")

    st.pyplot(plt)

    # Visualisasi Jumlah Permintaan Barang per Toko tiap jam
    st.subheader("Visualisasi Jumlah Permintaan Barang per Toko tiap Hari")
    st.write('Dataframe yang sudah dipreprocessing menggunakan model apriori menghasilkan 10 aturan asosiasi dari data yang sudah diolah dan ditampilkan di bawah ini : ')

    data_perhour = data.groupby('hour')['Transaction'].count()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=data_perhour.index ,
        y=data_perhour.values ,
                color="#FF0000"
    )
    plt.xticks(rotation=45)
    plt.title("Total Transaksi Perjam")
    plt.xlabel("Hour" , size = 15)
    plt.ylabel("Jumlah Transaksi")

    st.pyplot(plt)

if selected == "Rules":

    # Panggil fungsi apriori_rule()
    result_rules = apriori_rule()

    # Tampilkan hasil di Streamlit
    st.subheader("Top 10 Association Rules:")
    st.write('Dataframe yang digunakan sudah melalui tahap preprocessing dan sudah menggunakan model apriori untuk menghasilkan 10 association rules dari data yang sudah diolah dan di pakai menggunakan model algoritma apriori')
    st.dataframe(result_rules)


if selected == "Description":

    st.subheader("Analisis Keranjang Belanja dengan Algoritma Apriori")

    st.write(
        "Selamat datang di aplikasi web Analisis Keranjang Belanja (MBA)! Aplikasi ini menggunakan algoritma Apriori untuk menganalisis data transaksi "
        "dan menemukan asosiasi antara item yang sering dibeli bersama. Analisis Keranjang Belanja adalah teknik yang sangat berguna "
        "dalam dunia ritel dan e-commerce untuk memahami perilaku pembelian pelanggan, mengoptimalkan penempatan produk, dan meningkatkan strategi pemasaran."
    )

    st.markdown("### Algoritma Apriori:")
    st.write(
        "Algoritma Apriori adalah algoritma populer dalam data mining untuk menemukan itemset yang sering muncul dalam basis data transaksi. "
        "Algoritma ini beroperasi berdasarkan prinsip bahwa jika suatu itemset sering, maka semua subsetnya juga harus sering. Algoritma ini melibatkan "
        "langkah-langkah iteratif untuk menemukan itemset yang sering dan menghasilkan aturan asosiasi berdasarkan parameter yang ditentukan pengguna seperti support, confidence, "
        "dan lift."
    )

    st.markdown("### Dataset:")
    st.write(
        "Dataset yang digunakan dalam aplikasi ini berisi catatan transaksi dari lingkungan ritel. Setiap catatan mencakup informasi "
        "tentang transaksi, seperti tanggal, waktu, item yang dibeli, dan detail toko. Dataset tersebut telah diproses sebelumnya untuk mengekstrak fitur yang relevan, "
        "seperti bulan, hari, dan jam setiap transaksi, yang kemudian digunakan untuk analisis."
    )

    st.markdown("### Aplikasi MBA Apriori:")
    st.write(
        "Aplikasi web ini memungkinkan Anda untuk secara interaktif menjelajahi dan menganalisis data transaksi menggunakan algoritma Apriori. Berikut adalah gambaran singkat "
        "tentang fitur utama aplikasi:"
    )

    st.markdown("- **Beranda:** Pilih kriteria tertentu seperti item, bulan, dan hari untuk melihat data transaksi. Aplikasi menggunakan Apriori untuk menghasilkan aturan asosiasi dan rekomendasi berdasarkan item yang dipilih.")

    st.markdown("- **Visualisasi:** Lihat visualisasi 10 item paling banyak diminta dan total jumlah transaksi per toko setiap bulan. Pahami popularitas item dan tren permintaan dari waktu ke waktu.")

    st.markdown("- **Aturan:** Telusuri 10 aturan asosiasi teratas yang dihasilkan oleh algoritma Apriori. Temukan pola dan hubungan antara item yang cenderung dibeli bersama.")

    st.markdown("- **Deskripsi:** Baca penjelasan rinci tentang aplikasi Analisis Keranjang Belanja, dataset yang digunakan, dan algoritma Apriori.")
