import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer
from huggingface_hub import hf_hub_download
import pickle

# =================================================================
# ⚙️ CONFIG & RESOURCE LOADING
# =================================================================
HF_EMBEDDING_MODEL_REPO_ID = "Hanan-Alnakhal/solar-rag-embedding-model"
HF_LLM_MODEL_REPO_ID = "Hanan-Alnakhal/solar-rag-llm"

@st.cache_resource(show_spinner="Initializing Gaza Solar AI...")
def load_system():
    embed_model = SentenceTransformer(HF_EMBEDDING_MODEL_REPO_ID)
    qa_pipe = pipeline("text-generation", model=HF_LLM_MODEL_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL_REPO_ID)

    try:
        f_path = hf_hub_download(repo_id=HF_EMBEDDING_MODEL_REPO_ID, filename="my_solar_index.faiss")
        d_path = hf_hub_download(repo_id=HF_EMBEDDING_MODEL_REPO_ID, filename="my_solar_index_documents.pkl")
        index = faiss.read_index(f_path)
        with open(d_path, 'rb') as f: docs = pickle.load(f)
    except:
        docs = ["Solar panels in Gaza require regular cleaning due to dust.", "Standard consumption is 5-10kWh daily."]
        index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())
        index.add(np.array(embed_model.encode(docs)).astype('float32'))

    return embed_model, qa_pipe, tokenizer, index, docs

embed_m, qa_p, tk, f_index, kb_docs = load_system()

# =================================================================
# 🤖 FSM & SESSION INITIALIZATION
# =================================================================
st.set_page_config(page_title="Gaza Solar SEMS", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm your Solar Assistant. To give you better insights, **how many solar panels do you have in your house?**"}]
    st.session_state.stage = "ask_panels"  # Stages: ask_panels -> ask_consumption -> chat
    st.session_state.user_data = {"panels": 0, "consumption": 0}

# =================================================================
# 💬 CHAT DISPLAY
# =================================================================
st.title("☀️ Solar Management System")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =================================================================
# 🧠 LOGIC ENGINE
# =================================================================
if prompt := st.chat_input("Type here..."):
    # 1. Store and Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # Forces UI update to show user message immediately

# This handles the response generation AFTER the rerun
if st.session_state.messages[-1]["role"] == "user":
    user_input = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        response = ""

        # FSM STATE: Getting Panels
        if st.session_state.stage == "ask_panels":
            st.session_state.user_data["panels"] = user_input
            response = f"Got it, **{user_input} panels**. Now, what is your **average daily electricity consumption** (in kWh)?"
            st.session_state.stage = "ask_consumption"

        # FSM STATE: Getting Consumption
        elif st.session_state.stage == "ask_consumption":
            st.session_state.user_data["consumption"] = user_input
            response = "Thank you! I have saved your system details. You can now ask me any questions about solar maintenance or efficiency."
            st.session_state.stage = "chat"

        # FSM STATE: General RAG Chat
        else:
            with st.spinner("Analyzing knowledge base..."):
                q_emb = embed_m.encode([user_input]).astype('float32')
                _, I = f_index.search(q_emb, k=1)
                context = kb_docs[I[0][0]]

                # Context includes user's specific house data
                u_data = st.session_state.user_data
                full_prompt = f"User has {u_data['panels']} panels and uses {u_data['consumption']}kWh/day. Context: {context}\nQuestion: {user_input}\nAnswer:"

                raw_out = qa_p(full_prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']

                # FIX: Remove the prompt from the answer
                if "Answer:" in raw_out:
                    response = raw_out.split("Answer:")[-1].strip()
                else:
                    response = raw_out.replace(full_prompt, "").strip()

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})