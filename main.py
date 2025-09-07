import os
import asyncio
import streamlit as st
from cag.cag_agent import CagAgent
from rag.rag_agent_func import rag, rag_insert_data_to_db, rag_retrieve
from llmaindex.llma_index_agent import LmmaIndexAgent

# For Google Auth and Supabase
from st_supabase_connection import SupabaseConnection, execute_query
 # Remove incorrect import; use st.connection instead
import pandas as pd
import datetime


def login_screen():
    st.header("Welcome to PTI Chatbot")
    st.write("A chatbot for the Petroleum Training Institute")
    st.subheader("We have two modes:")
    st.button("Log in with Google", on_click=st.login)
    st.button("Use Public", on_click=lambda: set_mode("public"))

def set_mode(mode):
    st.session_state.mode = mode

def use_public():
    # --- Public Chat (no login) ---
    st.title("PTI Chatbot")
    st.caption("A chatbot for the Petroleum Training Institute")
    login_col = st.sidebar
    with login_col:
        st.button("Log In", on_click=st.login)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask your question about PTI Nigeria"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("In progress...", show_time=True):
                llmaIndexAgent =  LmmaIndexAgent(prompt, st.session_state.messages)
                response = llmaIndexAgent.rag_response
                context = llmaIndexAgent.llma_index_context
                answer = llmaIndexAgent.llma_index_answer
                print(f"Answer: {response}")
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main():

    st.set_page_config(
        page_title="PTI chatbot",
        page_icon="assets/pti_logo_bg.jpeg",
        menu_items={
            'Get Help': 'https://pti.edu.ng',
            'Report a bug': "https://wa.me/2348074974263",
            'About': "PTI Chatbot - A chatbot for the Petroleum Training Institute"
        }
    )
    st.logo("assets/pti_logo_bg.jpeg")
    st.html("<title>PTI chatbot</title>")
    st.html(hide_streamlit_watermark())

    # st.write(st.secrets)

    # --- Google Auth ---
    st.sidebar.write("Hi, Welcome 👋")
    # st.sidebar.markdown("---")
    # st.sidebar.subheader("Login for Private Chat")

    # Configure Google Auth (see https://docs.streamlit.io/develop/tutorials/authentication/google)
    # Set default mode if not set
    if "mode" not in st.session_state:
        st.session_state.mode = "login"

    # If user is not logged in and mode is login, show login screen
    if not st.user.is_logged_in and st.session_state.mode == "login":
        login_screen()
        return

    # If mode is public, show public chat
    if st.session_state.mode == "public":
        use_public()
        return

    # Otherwise, show private chat (user is logged in)
    login_col, public_col = st.sidebar.columns(2)
    with login_col:
        st.button("Log out", on_click=st.logout)
    with public_col:
        st.button("Use Public", on_click=lambda: set_mode("public"))

    name = getattr(st.user, 'name', None)
    username = getattr(st.user, 'username', None)
    email = getattr(st.user, 'email', None)
    authentication_status = True  # If st.user exists, user is authenticated
    if authentication_status:
        st.sidebar.success(f"Logged in as {name}")
        # st.sidebar.markdown("---")
        # st.sidebar.subheader("Your Private Chat History")

        # --- Supabase connection ---
        supabase = st.connection("supabase", type=SupabaseConnection)
        user_id = email or username or name or "unknown"

        # Load chat history from Supabase
        if "private_messages" not in st.session_state:
            # Try to fetch from Supabase
            try:
                df = execute_query(supabase.table("chat_history").select("*").eq("user_id", (user_id)), ttl=0)

                data = df.data if df.data else []
                if data:
                    df_data = pd.DataFrame(data)
                    st.session_state.private_messages = df_data[["role", "content"]].to_dict("records")
                else:
                    st.session_state.private_messages = []
            except Exception as e:
                st.session_state.private_messages = []

        # Show chat history in sidebar
        for msg in st.session_state.private_messages:
            # st.sidebar.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")
            pass

        st.title("PTI Private Chatbot")
        st.caption("A private chatbot for the Petroleum Training Institute (Google Authenticated)")

        # Show chat messages in main pane
        for message in st.session_state.private_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your private question about PTI Nigeria"):
            # Prepare user message
            user_msg = {"role": "user", "content": prompt}
            st.session_state.private_messages.append(user_msg)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("In progress...", show_time=True):
                    llmaIndexAgent =  LmmaIndexAgent(prompt, st.session_state.private_messages)
                    response = llmaIndexAgent.rag_response
                    context = llmaIndexAgent.llma_index_context
                    answer = llmaIndexAgent.llma_index_answer
                    print(f"Answer: {response}")
                    st.markdown(answer)
            # Prepare assistant message
            assistant_msg = {"role": "assistant", "content": answer}
            st.session_state.private_messages.append(assistant_msg)

            # Save both user and assistant messages to Supabase
            try:
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                rows = [
                    {"user_id": user_id, "role": "user", "content": str(prompt), "timestamp": str(timestamp)},
                    {"user_id": user_id, "role": "assistant", "content": str(answer), "timestamp": str(timestamp)}
                ]
                df2 = execute_query(
                    supabase.table("chat_history").insert(rows, count="None"),
                    ttl=0,
                )
                # print(df2)
            except Exception as e:
                st.warning("Could not save chat to database.")



def hide_streamlit_watermark():
    st.html("<style> ._container_gzau3_1, ._viewerBadge_nim44_23, ._profileContainer_gzau3_53 { display: none !important; } </style>")
    return "<script> const elementsToHide = [...document.querySelectorAll('._container_gzau3_1, ._viewerBadge_nim44_23, ._profileContainer_gzau3_53')]; elementsToHide.forEach(element => { element.style.display = 'none'; }); </script>"



if __name__ == "__main__":
    main()


