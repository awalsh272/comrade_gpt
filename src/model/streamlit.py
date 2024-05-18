import streamlit as st

from models import get_llm_response

st.title("Comrade GPT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about leftist theory!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container

    with st.chat_message("assistant"):
        stream = get_llm_response(prompt)
        response = st.write_stream(stream) #st.write(stream) #st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


# TODO:
# take previous q/a into account for context. Langchain has a doc for this
# deploy to AWS, use ECR and Fargate probably
# investigate if this will be expensive
