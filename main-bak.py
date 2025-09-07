import os
import asyncio
import streamlit as st
from cag.cag_agent import CagAgent
from rag.rag_agent_func import rag, rag_insert_data_to_db, rag_retrieve
# from rag.rag_agent import RagAgent
from llmaindex.llma_index_agent import LmmaIndexAgent


def main():
    st.set_page_config(
        page_title="PTI chatbot",
        page_icon="assets/pti_logo_bg.jpeg",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    st.logo("assets/pti_logo_bg.jpeg")
    st.sidebar.write("Hi, Welcome 👋")
    st.html("<title>PTI chatbot</title>")
    st.html(hide_streamlit_watermark())

    st.title("PTI Chatbot")
    st.caption("A chatbot for the Petroleum Training Institute")
    # st.subheader("A chatbot for the Petroleum Training Institute")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question about PTI Nigeria"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("In progress...", show_time=True):

                # cagAgent =  CagAgent(prompt)
                # response = cagAgent.cag_response

                # init = rag()
                # response = rag_retrieve(init, prompt, st.session_state.messages)

                llmaIndexAgent =  LmmaIndexAgent(prompt, st.session_state.messages)
                response = llmaIndexAgent.rag_response

                context = llmaIndexAgent.llma_index_context
                answer = llmaIndexAgent.llma_index_answer

                # print(f"Context: {context}\n Answer: {answer}")
                print(f"Answer: {response}")

                st.markdown(answer)
            # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})


def hide_streamlit_watermark():
    st.html("<style> ._container_gzau3_1, ._viewerBadge_nim44_23, ._profileContainer_gzau3_53 { display: none !important; } </style>")
    return "<script> const elementsToHide = [...document.querySelectorAll('._container_gzau3_1, ._viewerBadge_nim44_23, ._profileContainer_gzau3_53')]; elementsToHide.forEach(element => { element.style.display = 'none'; }); </script>"




if __name__ == "__main__":
    main()


