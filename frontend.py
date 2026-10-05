import streamlit as st
import backend as demo

st.set_page_config(
    page_title="Kenexai HR Handbook",
    page_icon="🎯",
    layout="centered",
)

st.title("📚 Kenexai HR Handbook")
st.caption("Powered by OpenAI + LangChain (few-shot) + Pinecone")

with st.sidebar:
    
    st.image("kenexai_logo.png", use_container_width=True)
    st.header("Controls")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory = demo.demo_memory()
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "**About this bot**\n\n"
        "By Ankita Malviya"
    )

if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Ask me anything...")
if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, updated_memory = demo.demo_conversation(
                input_text=input_text,
                memory=st.session_state.memory,
            )
        st.markdown(response)

    st.session_state.memory = updated_memory
    st.session_state.chat_history.append({"role": "assistant", "text": response})