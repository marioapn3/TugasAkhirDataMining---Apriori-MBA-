import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import association_rules, apriori
import calendar

# Load Dataset
df = pd.read_csv('DataSets.csv')
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday

df['month'].replace([i for i in range(1, 12 + 1)],
                   ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                   inplace=True)
df['day'].replace([i for i in range(6 + 1)],
                 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                 inplace=True)

st.title("Market Basket Analysis Invoice Sayur Menggunakan Algoritma Apriori")

def get_data(period_day='', weekday_weekend='', month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data['month'].str.contains(month.title())) &
        (data['day'].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else 'No Result'

def user_input_features():
    item = st.selectbox('Item', df['Item'].unique())
    month = st.select_slider('Month', df['month'].unique())
    day = st.select_slider('Day', df['day'].unique(), value="Saturday")
    return item, month, day

item, month, day = user_input_features()
data = get_data(month, day)

def hot_encode(x):
    return 1 if x >= 1 else 0

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

def parse_list(x):
    x = list(x)
    return x[0] if len(x) == 1 else ", ".join(x)

if type(data) == type("No Result"):
     st.error(data)

def return_item_df(items_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)
    
    filtered_data = data.loc[data["antecedents"] == items_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return None

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
st.sidebar.markdown("LinkedIn: https://www.linkedin.com/in/mario-aprilnino-625557217")
st.sidebar.markdown("Personal Website: https://marioapn3.github.io/")


st.sidebar.header("Source Code", anchor="center")
st.sidebar.markdown("You can see the source code at:")
st.sidebar.markdown("Github: ")

# Panggil fungsi apriori_rule()
result_rules = apriori_rule()

# Tampilkan hasil di Streamlit
st.subheader("Top 10 Association Rules:")
st.write('Dataframe yang digunakan untuk visualisasi adalah 10 association rules dari data yang sudah diolah dan di pakai menggunakan model algoritma apriori')
st.dataframe(result_rules)


st.subheader("Data 10 Item Paling Banyak Permintaan dari Toko")
st.write('Dataframe yang digunakan untuk visualisasi adalah 10 item yang paling banyak di request / diminta oleh setiap toko Transmart kepada VeegeFresh yang ada di Indonesia')

# Buat plot
plt.figure(figsize=(13, 5))
sns.set_palette('muted')
sns.barplot(x=data["Item"].value_counts()[:10].index,
            y=data["Item"].value_counts()[:10].values)
plt.xlabel("")
plt.ylabel("")
plt.xticks(size=13, rotation=45)
plt.title("10 Item paling laris")

# Tampilkan plot di Streamlit
st.pyplot(plt)

st.subheader("Visualisasi Jumlah Permintaan Barang per Toko tiap Bulan")
st.write('Dataframe yang digunakan untuk visualisasi adalah jumlah permintaan barang per toko tiap bulan yang request / diminta oleh setiap toko Transmart kepada VeegeFresh yang ada di Indonesia')

data_perbulan = data.groupby('month')['Transaction'].count()

# Convert month names to indices
month_indices = [list(calendar.month_name).index(month) for month in data_perbulan.index]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=[calendar.month_name[i] for i in month_indices],
    y=data_perbulan.values,
    color="#D5AAD3"
)
plt.xticks(rotation=-30, size=12)
plt.title("Jumlah transaksi per toko tiap bulan")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Transaksi")

st.pyplot(plt)
