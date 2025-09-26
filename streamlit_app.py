import streamlit as st
import time
from datetime import datetime
from dynamic_ai_chatbot import smart_context_aware_response


st.set_page_config(page_title="Dynamic AI Chatbot", layout="wide")

# Initialize theme state
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

col1, col2 = st.sidebar.columns([1,1])

# Define distinct styles for active and inactive buttons
active_style = """
    background-color: #00adb5;
    color: white;
    font-weight: bold;
    width: 100%;
    border-radius: 8px;
    padding: 5px 0;
    border: none;
"""
inactive_style = """
    background-color: #f0f0f0;
    color: black;
    width: 100%;
    border-radius: 8px;
    padding: 5px 0;
    border: 1px solid #ccc;
"""

with col1:
    # Render Light button with appropriate style
    if st.button("Light", key="light"):
        st.session_state.theme = "Light"
    # Use st.markdown hack to style the last button, requires matching key to button key
    st.markdown(f"""
    <style>
    div.stButton button:focus, div.stButton button:hover {{
        outline: none;
        cursor: pointer;
    }}
    div.stButton button#{st.session_state.theme.lower()} {{
        {active_style}
    }}
    </style>
    """, unsafe_allow_html=True)

with col2:
    if st.button("Dark", key="dark"):
        st.session_state.theme = "Dark"
    # Same styling hack here
    st.markdown(f"""
    <style>
    div.stButton button:focus, div.stButton button:hover {{
        outline: none;
        cursor: pointer;
    }}
    div.stButton button#{st.session_state.theme.lower()} {{
        {active_style}
    }}
    </style>
    """, unsafe_allow_html=True)

    theme = st.session_state.theme


# Set up CSS for both themes
if theme == "Dark":
    st.markdown("""
    <style>
    body, [data-testid="stAppViewContainer"], section.main > div {
        background-color: #222831 !important;
    }
    .chat-message {
        background: #3c434c !important;
        color: #fff !important;
        border: 2px solid #00adb5 !important;
        font-size: 1.75rem !important;
    }
    .user-message {
        background: linear-gradient(90deg, #495057 80%, #343b43 100%) !important;
        color: #fff !important;
        border: 2px solid #00adb5 !important;
        font-size: 1.75rem !important;
    }
    .bot-message {
        background: linear-gradient(90deg, #576474 80%, #37435a 100%) !important;
        color: #fff !important;
        border: 2px solid #00adb5 !important;
        font-size: 1.75rem !important;
    }
    .avatar {
        border: 2px solid #00adb5 !important;
        background: #3c434c !important;
    }
    .stTextInput > div > input {
        background-color: #434b54 !important;
        color: #fff !important;
        border: 2px solid #00adb5 !important;
    }
    .timestamp {
        color: #f8fafc !important;
        font-weight: bold !important;
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #fff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #222831 !important;
        color: #fff !important;
    }
    button, .stButton>button, .stDownloadButton>button {
        background: #fff !important;
        color: #222831 !important;
        border: 2px solid #00adb5 !important;
        font-weight: bold !important;
    }
    label, .stTextInput label {
        color: #fff !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
                
    /* Prevent cursor and editing in the selectbox */
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


else:
    st.markdown("""
    <style>
    /* Existing light theme CSS */
    body, [data-testid="stAppViewContainer"] {
        background-color: #ADD8E6;
    }
    section.main > div {
        background-color: #ADD8E6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(107, 155, 205, 0.15);
        max-width: 85%;
        word-wrap: break-word;
        font-size: 1.75rem;
        color: #141414;
    }
    .user-message {
        background: linear-gradient(90deg, #ffe6f0 80%, #fff0f9 100%);
        margin-right: auto;
        border-top-left-radius: 0.3rem;
        display: flex;
        align-items: flex-start;
    }
    .bot-message {
        background: linear-gradient(90deg, #d6f0d8 80%, #f4faf3 100%);
        margin-left: auto;
        border-top-right-radius: 0.3rem;
        display: flex;
        align-items: flex-start;
    }
    .avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        margin-right: 0.8rem;
        object-fit: cover;
        margin-top: 5px;
        border: 2px solid #cbe1f3;
        background: white;
    }
    .stTextInput > div > input {
        background-color: #e3f6ff !important;
        border-radius: 1rem !important;
        font-size: 1.25rem !important;
        padding: 0.7rem !important;
        color: #141414;
    }
    .bubble-content {
        flex: 1;
    }
    .timestamp {
        font-size: 0.85rem;
        color: #575757;
        margin-top: 0.2rem;
        position: relative;
        cursor: help;
    }
    .timestamp:hover::after {
        content: attr(title);
        position: absolute;
        background-color: #333;
        color: white;
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        white-space: nowrap;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        opacity: 0.9;
    }
    .chat-message, .chat-message * {
        font-size: 1.75rem !important;
        color: #141414 !important;
        line-height: 1.5;
    }
    [data-testid="stSidebar"] {
        background-color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)


# Custom CSS for the app styling and timestamp hover tooltip
st.markdown("""
<style>
/* General background color */
body, [data-testid="stAppViewContainer"] {
    background-color: #ADD8E6;
}
section.main > div {
    background-color: #ADD8E6;
}
/* Chat message styling */
.chat-message {
    padding: 1rem;
    border-radius: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(107, 155, 205, 0.15);
    max-width: 85%;
    word-wrap: break-word;
    font-size: 1.75rem;
    color: #141414; /* dark text for better contrast */
}
/* User messages */
.user-message {
    background: linear-gradient(90deg, #ffe6f0 80%, #fff0f9 100%);
    margin-right: auto;
    border-top-left-radius: 0.3rem;
    display: flex;
    align-items: flex-start;
}
/* Bot messages */
.bot-message {
    background: linear-gradient(90deg, #d6f0d8 80%, #f4faf3 100%);
    margin-left: auto;
    border-top-right-radius: 0.3rem;
    display: flex;
    align-items: flex-start;
}
/* Avatar styling */
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    margin-right: 0.8rem;
    object-fit: cover;
    margin-top: 5px;
    border: 2px solid #cbe1f3;
    background: white;
}
/* Input box styling */
.stTextInput > div > input {
    background-color: #e3f6ff !important;
    border-radius: 1rem !important;
    font-size: 1.25rem !important;
    padding: 0.7rem !important;
    color: #141414;
}
.bubble-content {
    flex: 1;
}
/* Timestamp */
.timestamp {
    font-size: 0.85rem;
    color: #575757;
    margin-top: 0.2rem;
    position: relative;
    cursor: help;
}
.timestamp:hover::after {
    content: attr(title);
    position: absolute;
    background-color: #333;
    color: white;
    padding: 3px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    white-space: nowrap;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    opacity: 0.9;
}
            
.chat-message, .chat-message * {
    font-size: 1.75rem !important;
    color: #141414 !important;
    line-height: 1.5;
            
[data-testid="stSidebar"] {
    background-color: #ff4b4b; /* Or any color you prefer */
}

</style>
""", unsafe_allow_html=True)


def current_time():
    return datetime.now().strftime("%H:%M")


# Clear chat history button in sidebar
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []

import json

if st.session_state.chat_history:
    # Convert to JSON so it's downloadable and readable
    chat_data = [
        {"user": user_msg, "bot": bot_msg, "time": msg_time}
        for (user_msg, bot_msg, msg_time) in st.session_state.chat_history
    ]
    chat_json = json.dumps(chat_data, indent=2)
    st.sidebar.download_button(
        label="Download Chat History",
        data=chat_json,
        file_name="chat_history.json",
        mime="application/json",
    )


st.markdown("<h1 style='text-align:center; color:#003366;'>🤖 Dynamic AI Chatbot</h1>", unsafe_allow_html=True)


# Display chat messages with avatars and timestamps
for i, (user_msg, bot_msg, msg_time) in enumerate(st.session_state.chat_history):
    # User message bubble on left
    st.markdown(
        f"""
        <div class="chat-message user-message">
            <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" alt="User">
            <div class="bubble-content">
            <b>You:</b> {user_msg}
                <div class="timestamp" title="{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}">{msg_time}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bot message bubble on right
    st.markdown(
        f"""
        <div class="chat-message bot-message">
            <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" alt="Bot">
            <div class="bubble-content">
                <b>Bot:</b> {bot_msg}
                <div class="timestamp" title="{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}">{msg_time}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Only ONE text_input widget, with a unique key
user_input = st.text_input("💬 Type your message here:", key="user_input")
send = st.button("Send")

if send and user_input:
    with st.spinner("🤖 Bot is typing..."):
        response = smart_context_aware_response(user_input, st.session_state.chat_history)
        time.sleep(0.5)
    st.session_state.chat_history.append((user_input, response, current_time()))
    st.rerun()  # If available, otherwise use st.experimental_rerun()





