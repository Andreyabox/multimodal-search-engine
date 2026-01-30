import streamlit as st


st.set_page_config(
    page_title="Умный поиск картинок",
    layout="wide"
)

st.title("Мультимодальный поиск картинок")
tab1, tab2 = st.tabs(["Поиск картинок", "Загрузка картинок"])

with tab1:
    st.header("Поиск картинок")
    query = st.text_input("Введите запрос для поиска")

with tab2:
    st.header("Загрузка картинок")
    st.write("Загрузите ваши картинки для последующего поиска по ним.")

with st.sidebar:
    st.title("О приложении")
    st.write("Это приложение для поиска картинок по текстовому запросу")