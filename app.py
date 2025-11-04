import streamlit as st
import os
import sys
from rag_agent_hf import get_agent_instance # Import the main function

# --- UI SETUP ---
st.set_page_config(page_title="LangGraph RAG Agent", layout="wide")
st.title("🧠 LangGraph RAG Agent (Hugging Face Model)")
st.markdown("Ask a question related to the documents in the `knowledge_base/` folder to test Retrieval-Augmented Generation (RAG).")
st.divider()

# --- AGENT INITIALIZATION ---

@st.cache_resource
def load_agent():
    """Load the LangGraph agent once and cache it."""
    try:
        # Suppress command line print statements in Streamlit's output
        with st.spinner("Initializing LLM (Mistral 7B) and VectorDB (ChromaDB)... This may take a moment."):
            agent = get_agent_instance()
            return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}. Check your hardware and model downloads.")
        return None

rag_agent_app = load_agent()

if rag_agent_app is None:
    st.stop()
    
# --- CHAT INTERFACE ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am a RAG Agent. What would you like to know?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question..."):
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process query with the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Define the initial state for the LangGraph invocation
        initial_state = {"question": prompt, "answer": "", "context": [], "action": ""}

        # Run the agent (no streaming in this basic version, invoke is simpler)
        try:
            # Use app.invoke() to run the full graph
            final_state = rag_agent_app.invoke(initial_state)
            
            # Extract relevant information
            answer = final_state['answer']
            reflection_status = final_state.get('reflection_status', 'N/A')
            context = final_state.get('context', [])
            
            # Format the output for display
            full_response = f"**Answer:**\n\n{answer}\n\n---\n\n"
            full_response += f"**Agent Reflection:** {reflection_status}\n\n"
            if context and context[0] != "Error: Retrieval skipped, DB not ready.":
                full_response += f"**Retrieved Context Snippet:**\n- {context[0][:150]}...\n\n"
            elif final_state.get('action') == 'DIRECT_ANSWER':
                full_response += "*Retrieval was skipped by the Plan node (Direct Answer).* \n"
            
            message_placeholder.markdown(full_response)
        
        except Exception as e:
            error_message = f"An error occurred during agent execution: {e}"
            message_placeholder.error(error_message)
            full_response = error_message
        
    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})