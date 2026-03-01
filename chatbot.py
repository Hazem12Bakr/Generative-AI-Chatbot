from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.memory import ConversationBufferWindowMemory
import os

# Load the environment variables from the .env file
# We use the load_dotenv function to load the environment variables from the .env file. This allows us to access the GROQ_API_KEY variable
# in our code using os.getenv('GROQ_API_KEY').

#This is because load_dotenv() loads the environment variables into the environment, making them accessible as global variables in our code.
load_dotenv()

# Streamlit page setup | it must be the first part of the code to ensure that the page configuration
# is set before any other Streamlit components are rendered.
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("💬Generative AI Chatbot")
st.subheader("By Hazem Abo Bakr")

user_prompt = st.chat_input("💬 How may I assist you today...")

# Intiate chat history
# This code will be run only once when the app is first loaded.
# It checks if the "memory" key is not already in the session state, and if it's not, it initializes it as an empty list.
# This allows us to store the conversation history between the user and the chatbot across different interactions.
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k = 5,
        return_messages = True
    )


# Show the chat history
# This code iterates through the memory chat history stored in the momory and displays each message in the chat interface using st.chat_message.
# The role of the message (either "user" or "assistant") is determined by the message's role field in the chat history.
for message in st.session_state.memory.chat_memory.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# llm intiate
with st.sidebar:
    st.header("⚙️ Settings")
    # The slider allows the user to adjust the temperature parameter, which controls the randomness of the model's responses.
    # A lower temperature will result in more deterministic responses, while a higher temperature will produce more varied and creative responses.
    temperature = st.slider("Response Creativity", 0.0, 0.1, 0.2, 0.1)
    # The button allows the user to clear the chat history. When the button is clicked, it clears the memory and reruns the app to reflect the changes.
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.rerun()

llm = ChatGroq(
    model = 'llama-3.1-8b-instant',
    temperature = temperature
)


if user_prompt:

    # Display user message in chat interface
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Add user message to memory
    st.session_state.memory.chat_memory.add_user_message(user_prompt)

    # Prepare messages (with system prompt)
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in clear, structured and beginner-friendly explanations."},
        *st.session_state.memory.chat_memory.messages
    ]

    # Call the model and stream the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(messages):
            full_response += chunk.content or ""
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add AI response to memory
    st.session_state.memory.chat_memory.add_ai_message(full_response)




# if user_prompt:

#     # Display user message in chat interface
#     with st.chat_message("user"):
#         st.markdown(user_prompt)
    
#     # Add user message to memory
#     st.session_state.memory.chat_memory.add_user_message(user_prompt)

#     # Prepare messages (with system prompt)
#     messages = [
#         {"role": "system", "content": "You are an AI assistant specialized in clear, structured and beginner-friendly explanations."},
#         *st.session_state.memory.chat_memory.messages
#     ]

#     # Call the model
#     response = llm.invoke(messages)

#     assistant_response = response.content

#     # Add AI response to memory
#     st.session_state.memory.chat_memory.add_ai_message(assistant_response)

#     # Display assistant response
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)

    
