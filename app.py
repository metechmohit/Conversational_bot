import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch 

# Use st.cache_resource to load models only once
@st.cache_resource
def load_models():
    # --- T5 Expansion Model ---
    t5_model_name = "metechmohit/final_t5_expansion_model" 
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    
    # --- BERT Topic Model ---
    bert_model_name = "metechmohit/final_bert_topic_model"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)

    # Create pipelines
    expansion_pipeline = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer)
    classification_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)
    
    return expansion_pipeline, classification_pipeline

# --- Helper Function ---
def format_history(messages):
    """Formats the chat history into a single string for the T5 model."""
    if not messages:
        return ""
    
    # Join messages with ' | '
    history_str = " | ".join(
        f"{msg['role']}: {msg['content']}" for msg in messages
    )
    return history_str

# --- Streamlit App ---

st.title("🤖 Adaptive UPSC Chatbot")
st.caption("A demo of real-time query expansion and topic tagging.")

# Load models
try:
    expansion_pipeline, classification_pipeline = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models. Have you pushed them to the Hugging Face Hub and updated the model names in the code? \n\n{e}")
    st.stop()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display the extra info (expansion & topic) for assistant messages
        if message["role"] == "assistant":
            st.caption(f"Expanded Query: {message['expansion']}")
            st.caption(f"Predicted Topic: {message['topic']}")


# Get new user input
if prompt := st.chat_input("Ask about Indian Politics, History, etc."):
    
    # 1. Format History
    # Send only the last 20 messages (10 turns)
    history_str = format_history(st.session_state.messages[-20:])
    
    # Add user message to session state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get Model Predictions
    with st.spinner("Analyzing your query..."):
        # --- T5 Query Expansion ---
        t5_input = f"history: {history_str} query: {prompt}"
        expanded_query = expansion_pipeline(t5_input, max_new_tokens=256)[0]['generated_text']

        # --- BERT Topic Tagging ---
        # We classify the *expanded* query for better accuracy
        classification_result = classification_pipeline(expanded_query)
        predicted_topic = classification_result[0]['label']
        confidence = classification_result[0]['score']

    # 3. Formulate Bot Response and Display
    # For this assignment, the "answer" is just showing what the model understood.
    bot_response = f"Okay, I understand your request."

    # Add bot response to session state
    bot_message = {
        "role": "assistant", 
        "content": bot_response,
        "expansion": expanded_query, # Store extra info for display
        "topic": f"{predicted_topic} (Confidence: {confidence:.2f})" # Store extra info
    }
    st.session_state.messages.append(bot_message)
    
    # Display bot response and the "debug" info
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.caption(f"Expanded Query: {expanded_query}")
        st.caption(f"Predicted Topic: {predicted_topic} (Confidence: {confidence:.2f})")