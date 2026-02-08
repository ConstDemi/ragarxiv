import os
import streamlit as st
import requests

st.set_page_config(page_title="Science RAG", layout="wide")
st.title("arXiv Info System")

# === КОНФИГ ===
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/ask")
TIMEOUT = 600

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Sources (Top-K)", 1, 10, 3)

# История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Основная логика
if prompt := st.chat_input("Ask about scientific papers..."):
    if len(prompt) > 1000:
        st.warning("Question is too long (max 1000 chars)")
        st.stop()
    
    # 1. Отображаем вопрос
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Получаем данные
    response_data = None
    error_message = None

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            try:
                resp = requests.post(
                    BACKEND_URL, 
                    json={"text": prompt, "top_k": top_k}, 
                    timeout=TIMEOUT
                )
                
                if resp.status_code == 200:
                    response_data = resp.json()
                elif resp.status_code == 503:
                    error_message = "⏳ System is initializing. Please wait."
                else:
                    error_message = f"❌ Server Error: {resp.status_code}"
            
            except requests.exceptions.Timeout:
                error_message = "⏱️ Request timeout."
            except requests.exceptions.ConnectionError:
                error_message = "🔌 Backend is offline."
            except Exception as e:
                error_message = f"❌ Unexpected error: {e}"

            if error_message:
                st.error(error_message)
            
            elif response_data:
                answer = response_data.get("answer", "No answer provided.")
                sources = response_data.get("sources", [])
                
                # Основной ответ
                st.markdown(answer)
                
                # Источники
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {src.get('text', '').strip()}")
                            st.divider()
                
                # Метаданные
                st.caption(f"⏱️ {response_data.get('process_time', 0):.2f}s | 📄 {len(sources)} chunks")

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })