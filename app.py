import streamlit as st
import pandas as pd
import joblib
import os
import random
from groq import Groq

# Import from custom modules
from config import (
    GROQ_API_KEY, GROQ_MODEL_NAME, DATA_FILE_PATH,
    PREDICTIVE_MODEL_ASSETS_PATH, PERSONA_MODEL_ASSETS_PATH, RAG_PDF_PATHS
)
from src.strategy_playbook import strategy_playbook
from src.chatbot_engine import IntegratedChatbot
from src.rag_handler import load_rag_retriever

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Loan Collection Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- State Management ---
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'analysis' not in st.session_state:
    st.session_state.analysis = {}
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = {}
if 'current_customer_data' not in st.session_state:
    st.session_state.current_customer_data = None

# --- Load Assets (Cached) ---
@st.cache_resource
def load_all_assets():
    """
    Loads predictive, persona, and RAG assets required for the application.
    Only rebuilds RAG vectorstore if it doesn't exist or PDFs have changed.
    """
    with st.status("Initializing models and knowledge base...", expanded=True) as status:
        # --- Check model assets ---
        if not os.path.exists(PREDICTIVE_MODEL_ASSETS_PATH) or not os.path.exists(PERSONA_MODEL_ASSETS_PATH):
            status.update(label="Model assets not found. Please run the training scripts first.", state="error")
            st.error("Model files not found. Please run `scripts/train_predictive_model.py` and `scripts/train_persona_model.py`")
            return None, None, None

        st.write("Loading predictive model assets...")
        predictive_assets = joblib.load(PREDICTIVE_MODEL_ASSETS_PATH)
        st.write("Loading persona model assets...")
        persona_assets = joblib.load(PERSONA_MODEL_ASSETS_PATH)

        # --- RAG retriever / vectorstore check ---
        st.write("Setting up RAG retriever...")
        vectorstore_exists = os.path.exists("vectorstore_index") and os.path.exists("vectorstore_index/index.faiss")
        if vectorstore_exists:
            st.write("⏳ FAISS vectorstore exists. Loading from disk...")
            retriever = load_rag_retriever(RAG_PDF_PATHS)  # will load cached vectorstore
        else:
            st.write("⏳ FAISS vectorstore not found. Building from PDFs...")
            retriever = load_rag_retriever(RAG_PDF_PATHS)  # will build and save

        status.update(label="Initialization Complete!", state="complete")

    return predictive_assets, persona_assets, retriever


# --- Main App ---
st.title("🤖 Intelligent Loan Collection Chatbot")
st.markdown("This application uses a combination of a predictive model, a persona clustering model, and a RAG-powered LLM to conduct intelligent, empathetic, and compliant loan collection conversations.")

# --- Initialization ---
predictive_assets, persona_assets, retriever = load_all_assets()

if predictive_assets and persona_assets and retriever:
    if st.session_state.chatbot is None:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            st.session_state.chatbot = IntegratedChatbot(
                predictive_assets, persona_assets, strategy_playbook,
                groq_client, GROQ_MODEL_NAME, retriever
            )
            st.success("Chatbot is ready!")
        except Exception as e:
            st.error(f"Failed to initialize Groq client. Please check your API key. Error: {e}")

# --- UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("👤 Customer Profile Input")
    st.markdown("Select a customer profile to simulate a conversation. The models will predict default risk and persona, then initiate the chat.")

    # --- Customer Data Input ---
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        customer_id = st.selectbox("Select a Customer ID", df['CustomerID'].unique())

        # Initialize customer_data with the selected customer
        customer_data = df[df['CustomerID'] == customer_id].to_dict('records')[0]

        # Option to generate random customer
        if st.button("Generate Random Customer"):
            random_id = f"Random_{random.randint(1000,9999)}"
            customer_data = df.sample(1).to_dict('records')[0]
            customer_data['CustomerID'] = random_id
            # Store in session state to persist the random customer
            st.session_state.current_customer_data = customer_data
            st.success(f"Generated random customer: {random_id}")
            st.rerun()

        # Use session state customer data if it exists (for random customers)
        if st.session_state.current_customer_data is not None:
            customer_data = st.session_state.current_customer_data

        # Option to edit/create new customer
        with st.expander("✏️ Edit / Create New Customer"):
            editable_data = {}
            for k, v in customer_data.items():
                editable_data[k] = st.text_input(f"{k}", str(v))
                
            for col in ['Age', 'Income', 'LoanAmount', 'InterestRate', 'TenureMonths',
                'MissedPayments', 'DelaysDays', 'PartialPayments',
                'ResponseTimeHours', 'AppUsageFrequency', 'WebsiteVisits',
                'InteractionAttempts', 'Complaints', 'SentimentScore']:
                if col in editable_data:
                    try:
                        editable_data[col] = float(editable_data[col])
                    except ValueError:
                        editable_data[col] = 0.0 # default fallback
            
            if st.button("Apply Edits"):
                customer_data = editable_data
                st.session_state.current_customer_data = customer_data
                st.success("Customer data updated!")
                st.rerun()

        if st.button("Start Conversation", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer and initiating conversation..."):
                chatbot = st.session_state.chatbot
                opening_message, default_prob, persona = chatbot.start_conversation(customer_data)

                st.session_state.messages = [{"role": "assistant", "content": opening_message}]
                st.session_state.conversation_started = True
                st.session_state.analysis = {
                    "Default Probability": f"{default_prob:.2%}",
                    "Predicted Persona": persona
                }
                st.session_state.last_analysis = {}  # reset
                st.rerun() # Rerun to update the chat interface

        # Reset button to clear random customer
        if st.session_state.current_customer_data is not None:
            if st.button("Reset to Selected Customer"):
                st.session_state.current_customer_data = None
                st.rerun()

    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_FILE_PATH}. Please make sure it exists.")
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")

    # Display customer data after selection
    if 'customer_data' in locals():
        st.subheader("Selected Customer Data")
        st.json(customer_data, expanded=False)

    # --- Sidebar: Thinking Process ---
    st.sidebar.title("🧠 Chatbot's Brain")
    if st.session_state.conversation_started and st.session_state.chatbot:
        sidebar_analysis = st.sidebar.container()
        with sidebar_analysis:
            st.subheader("Initial Analysis")
            st.write(st.session_state.analysis)

            st.subheader("Key Predictive Features")
            pred_features = {k: v for k, v in st.session_state.chatbot.prediction_features.items() if isinstance(v, (int, float)) and abs(v) > 0.1}
            st.json(pred_features, expanded=False)

            st.subheader("Key Persona Features")
            pers_features = {k:v for k,v in st.session_state.chatbot.persona_features.items() if 'Score' in k or 'Capacity' in k or 'Reliability' in k}
            st.json(pers_features, expanded=False)
            
            st.subheader("Real-time Analysis")
            if st.session_state.last_analysis:
                analysis = st.session_state.last_analysis
                chatbot = st.session_state.chatbot
                
                # Display analysis data
                analysis_data = {
                    "Sentiment": analysis.get('sentiment', 'N/A'),
                    "User Intent": analysis.get('user_intent', 'N/A'),
                    "Strike Count": analysis.get('strike_count', f"{chatbot.negative_sentiment_strikes}/{chatbot.max_negative_strikes}"),
                    "Last Turn Score (%)": analysis.get('last_turn_score', "N/A"),
                    "Cumulative Score (%)": f"{analysis.get('cumulative_score', 0):.1f}" if 'cumulative_score' in analysis else "N/A",
                    "Current LLM Strategy": analysis.get('applied_strategy', "N/A")
                }
                
                st.sidebar.write(analysis_data)
                
                # Progress bar for cumulative score
                if 'cumulative_score' in analysis:
                    cumulative_score = analysis['cumulative_score']
                    st.sidebar.progress(max(0, min(100, int(cumulative_score))) / 100)
                    
                # Optional: Add color coding for strikes
                if chatbot.negative_sentiment_strikes >= 2:
                    st.sidebar.warning(f"⚠️ High strike count: {chatbot.negative_sentiment_strikes}/{chatbot.max_negative_strikes}")
                
                # Optional: Add conversation status
                if chatbot.conversation_ended:
                    st.sidebar.error("🔴 Conversation Ended")
                else:
                    st.sidebar.success("🟢 Conversation Active")
                    
            else:
                st.sidebar.write("No analysis yet.")

with col2:
    st.header("💬 Conversation")
    chat_container = st.container(height=600)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if st.session_state.conversation_started:
        if st.session_state.chatbot and st.session_state.chatbot.conversation_ended:
            st.warning("⚠️ This conversation has ended based on the chatbot's protocol (likely due to repeated negative sentiment).")
        else:
            if prompt := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with st.spinner("Thinking..."):
                    chatbot = st.session_state.chatbot
                    response, analysis = chatbot.get_next_response(prompt)
                    st.session_state.last_analysis = analysis

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    if st.session_state.chatbot and st.session_state.chatbot.conversation_ended:
        st.warning("⚠️ This conversation has ended based on the chatbot's protocol.")