import streamlit as st
import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


st.set_page_config(
    page_title="Умный поиск картинок",
    layout="wide"
)

st.title("Мультимодальный поиск картинок")
tab1, tab2 = st.tabs(["Поиск картинок", "Загрузка картинок"])

DEFAULT_API_BASE_URL = os.environ.get("SEARCH_API_BASE_URL", "http://localhost:8000")


def _search_images(api_base_url: str, query: str, top_k: int) -> dict:
    url = f"{api_base_url.rstrip('/')}/search"
    body = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        payload = response.read()

    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from API: {payload!r}") from exc


with tab1:
    st.header("Поиск картинок")

    if "api_base_url" not in st.session_state:
        st.session_state["api_base_url"] = DEFAULT_API_BASE_URL
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None
        st.session_state["search_query"] = None

    with st.form("search_form", border=False):
        query = st.text_input("Введите запрос для поиска на английском языке", value="car")
        top_k = st.number_input(
            "Количество результатов",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )
        api_base_url = st.text_input(
            "API base URL",
            value=st.session_state["api_base_url"],
            help="Например: http://localhost:8000",
        )
        submitted = st.form_submit_button("Найти", type="primary")

    st.session_state["api_base_url"] = api_base_url

    if submitted:
        trimmed_query = (query or "").strip()
        if not trimmed_query:
            st.warning("Введите текстовый запрос для поиска.")
        else:
            try:
                with st.spinner("Ищем..."):
                    response = _search_images(
                        api_base_url=api_base_url,
                        query=trimmed_query,
                        top_k=int(top_k),
                    )
                st.session_state["search_results"] = response.get("results", [])
                st.session_state["search_query"] = response.get("query", trimmed_query)
            except HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                st.error(f"API вернуло ошибку HTTP {exc.code}: {details}")
            except (URLError, socket.timeout, TimeoutError) as exc:
                st.error(f"Не удалось подключиться к API ({api_base_url}): {exc}")
            except Exception as exc:
                st.error(f"Ошибка поиска: {exc}")

    results = st.session_state.get("search_results")
    if results is not None:
        st.subheader(f"Результаты: {st.session_state.get('search_query') or ''}")
        if not results:
            st.info("Ничего не найдено.")
        else:
            for item in results:
                image_url = item.get("image_url")
                caption = item.get("caption")
                score = item.get("score")

                if image_url:
                    st.image(image_url, caption=f"{caption or ''} (score: {score})")
                else:
                    st.write({"caption": caption, "score": score, "image_url": image_url})

with tab2:
    st.header("Загрузка картинок")
    st.write("Загрузите ваши картинки для последующего поиска по ним.")

with st.sidebar:
    st.title("О приложении")
    st.write("Это приложение для поиска картинок по текстовому запросу")
