import os
import streamlit as st
from mistralai.client import Mistral

st.set_page_config(page_title="NK JS Chat", page_icon="💬")

SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_NAME = "mistral-large-latest"

def get_api_key() -> str:
    """Read the API key from Streamlit secrets first, then env var."""
    if "MISTRAL_API_KEY" in st.secrets:
        return st.secrets["MISTRAL_API_KEY"]
    return os.getenv("MISTRAL_API_KEY", "")

@st.cache_resource
def get_client(api_key: str) -> Mistral:
    return Mistral(api_key=api_key)

def main():
    st.title("💬 NK Job Check")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No API key found. Add MISTRAL_API_KEY to "
            ".streamlit/secrets.toml locally, or to the app's Secrets "
            "panel on Streamlit Community Cloud."
        )
        st.stop()

    client = get_client(api_key)

    task = "Provide list of job openings for Senior software developer for C#, ASP.Net, Azure, Devops, AI, Data, Cyber security"
    context = "in Hyderabad India location who are ready to provide work from home posted in the last 90 days having minimum salary of 4000000 lacs"
    returnFormat = "In Json format should have company name, Contact Details, Salary expectations, Employee Reviews"

    user_content = f"""
    {task} {context}, {returnFormat}
    """

    # Initialize conversation history with system prompt and default user content
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Get the assistant's response to the default user content
        with st.spinner("Generating default response..."):
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=st.session_state.messages,
            )
            assistant_reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Render all messages (skip the system message)
    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input for new messages
    user_input = st.chat_input("Anything else you want to search today...")
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