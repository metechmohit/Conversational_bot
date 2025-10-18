import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch 
import openai 

@st.cache_resource
def load_models():
    # T5 Expansion Model
    t5_model_name = "metechmohit/final_t5_expansion_model" 
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    
    # BERT Topic Model 
    bert_model_name = "metechmohit/final_bert_topic_model"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)

    # pipelines
    expansion_pipeline = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer)
    classification_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)
    
    return expansion_pipeline, classification_pipeline

def format_history(messages):
    if not messages:
        return ""
    history_str = " | ".join(
        f"{msg['role']}: {msg['content']}" for msg in messages
    )
    return history_str

def get_openai_response(api_key, query):
    try:
        openai.api_key = api_key
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # Efficient and cost-effective model
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert AI assistant for UPSC (Indian Civil Services Exam) aspirants. Answer the following question concisely and accurately, in 1-2 sentences."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            max_tokens=100, 
            temperature=0.7 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, an error occurred with the OpenAI API: {e}"


st.title("Adaptive UPSC Chatbot")
st.caption("This bot expands your query, tags it to a topic, and provides a concise answer.")

with st.sidebar:
    st.header("Configuration")
    st.markdown("Please enter your OpenAI API key to enable answers.")
    api_key_input = st.text_input(
        "OpenAI API Key", 
        type="password", 
        placeholder="sk-..."
    )
    st.markdown("---")


# Load models
try:
    expansion_pipeline, classification_pipeline = load_models()
    st.success("Local models loaded successfully!")
except Exception as e:
    st.error(f"Error loading Hugging Face models: {e}")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Expanded Query: {message['expansion']}")
            st.caption(f"Predicted Topic: {message['topic']}")

# Get new user input
if prompt := st.chat_input("Ask about Politics, History, etc."):
    
    # Add user message to session state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Expand and Classify Query
    with st.spinner("Analyzing your query..."):
        
        # CONTEXT WINDOW 
        history_messages = st.session_state.messages[-21:-1] # Get 10 previous messages (excluding current)
        history_str = format_history(history_messages)
        t5_input = f"history: {history_str} query: {prompt}"
        expanded_query = expansion_pipeline(t5_input, max_new_tokens=256)[0]['generated_text']
        
        classification_result = classification_pipeline(expanded_query)
        predicted_topic = classification_result[0]['label']
        confidence = classification_result[0]['score']

    # Generate the actual response
    bot_response = ""
    with st.spinner("Generating answer..."):
        if not api_key_input:
            bot_response = "Please enter your OpenAI API key in the sidebar to get an answer."
        else:
            bot_response = get_openai_response(api_key_input, expanded_query)

    # Add bot response to session state and display
    bot_message = {
        "role": "assistant", 
        "content": bot_response,
        "expansion": expanded_query, 
        "topic": f"{predicted_topic} (Confidence: {confidence:.2f})"
    }
    st.session_state.messages.append(bot_message)
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        st.caption(f"Expanded Query: {expanded_query}")
        st.caption(f"Predicted Topic: {predicted_topic} (Confidence: {confidence:.2f})")