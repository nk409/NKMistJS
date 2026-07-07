"""
Streamlit console-style chat app powered by the Mistral API.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io) --
see README.md for step-by-step instructions.
"""

import os
import streamlit as st
from mistralai.client import Mistral

st.set_page_config(page_title="Mistral Chat", page_icon="💬")

SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_NAME = "mistral-large-latest"


def get_api_key() -> str:
    """Read the API key from Streamlit secrets first, then env var.

    Locally: put it in .streamlit/secrets.toml (gitignored).
    On Streamlit Community Cloud: set it in the app's "Secrets" panel --
    it shows up the same way, via st.secrets.
    """
    if "MISTRAL_API_KEY" in st.secrets:
        return st.secrets["MISTRAL_API_KEY"]
    return os.getenv("MISTRAL_API_KEY", "")


@st.cache_resource
def get_client(api_key: str) -> Mistral:
    return Mistral(api_key=api_key)


def main():
    st.title("💬 Mistral Chat")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No API key found. Add MISTRAL_API_KEY to "
            ".streamlit/secrets.toml locally, or to the app's Secrets "
            "panel on Streamlit Community Cloud."
        )
        st.stop()

    client = get_client(api_key)

    # Keep conversation history across reruns (Streamlit reruns the whole
    # script on every interaction, so state must live in st.session_state).
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Render past turns (skip the system message)
    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Job Search Query...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.complete(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()


if __name__ == "__main__":
    main()
