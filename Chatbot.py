import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


# Page configuration
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sample FAQ data - you can replace this with your own data
FAQ_DATA = {
    "questions": [
        "What is your return policy?",
        "How can I contact customer support?",
        "What payment methods do you accept?",
        "How long does shipping take?",
        "Do you offer international shipping?",
        "What is your privacy policy?",
        "How can I track my order?",
        "Do you have a mobile app?",
        "What are your business hours?",
        "How can I reset my password?",
        "What if my item arrives damaged?",
        "Do you offer refunds?",
        "How can I create an account?",
        "What are your shipping costs?",
        "Do you have a loyalty program?"
    ],
    "answers": [
        "We offer a 30-day return policy for most items. Items must be in original condition with all tags attached. Please contact our customer service team to initiate a return.",
        "You can contact our customer support team via email at support@company.com, phone at 1-800-123-4567, or through our live chat feature on our website.",
        "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, Google Pay, and bank transfers for orders over $100.",
        "Standard shipping typically takes 3-5 business days within the continental US. Express shipping (1-2 business days) is available for an additional fee.",
        "Yes, we offer international shipping to most countries. Shipping times and costs vary by location. Please check our international shipping page for specific details.",
        "We are committed to protecting your privacy. We collect only necessary information to process your orders and improve our services. We never sell your personal information to third parties.",
        "You can track your order by logging into your account and visiting the 'My Orders' section, or by using the tracking number provided in your shipping confirmation email.",
        "Yes, we have a mobile app available for both iOS and Android devices. You can download it from the App Store or Google Play Store.",
        "Our customer service team is available Monday through Friday, 9 AM to 6 PM EST. Our online store is open 24/7 for your convenience.",
        "To reset your password, click on the 'Forgot Password' link on the login page. Enter your email address and follow the instructions sent to your email.",
        "If your item arrives damaged, please take photos of the damage and contact our customer service team within 48 hours of delivery. We'll arrange a replacement or refund.",
        "Yes, we offer refunds for items returned within our 30-day return window. Refunds are processed within 5-7 business days after we receive your return.",
        "You can create an account by clicking the 'Sign Up' button on our website. You'll need to provide your name, email address, and create a password.",
        "Standard shipping is free for orders over $50. For orders under $50, shipping costs $5.99. Express shipping costs $12.99 regardless of order value.",
        "Yes, we have a loyalty program called 'Rewards Plus'. Earn points on every purchase and redeem them for discounts on future orders. Sign up is free!"
    ]
}

GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

class FAQChatbot:
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.question_vectors = None
        self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on the FAQ questions"""
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def find_best_match(self, user_question, threshold=0.3):
        """Find the best matching FAQ question"""
        # Preprocess user question
        processed_question = self.preprocess_text(user_question)
        
        # Vectorize user question
        user_vector = self.vectorizer.transform([processed_question])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.question_vectors).flatten()
        
        # Find the best match
        best_match_idx = np.argmax(similarity_scores)
        best_score = similarity_scores[best_match_idx]
        
        if best_score >= threshold:
            return best_match_idx, best_score
        else:
            return None, best_score
    
    def get_response(self, user_question):
        """Get response for user question"""
        best_match_idx, score = self.find_best_match(user_question)
        
        if best_match_idx is not None:
            return {
                'answer': self.answers[best_match_idx],
                'confidence': score,
                'matched_question': self.questions[best_match_idx]
            }
        else:
            return {
                'answer': None,
                'confidence': score,
                'matched_question': None
            }

def main():
    # Initialize chatbot
    chatbot = FAQChatbot(FAQ_DATA['questions'], FAQ_DATA['answers'])
    
    # Header
    st.markdown('<h1 style="text-align:center; font-size:2.5rem;">🤖 FAQ Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:gray;">Select a question on the left or type your own.</div>', unsafe_allow_html=True)
    st.markdown('---')
    
    # FAQ buttons at the top
    st.markdown('<h3 style="margin-top:2em;">You can ask about these topics:</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    
    with col1:
        # Display FAQ buttons
        st.markdown('<h3>FAQs</h3>', unsafe_allow_html=True)
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''
        for i, question in enumerate(FAQ_DATA['questions']):
            if st.button(question, key=f"faq_{i}"):
                st.session_state.user_input = question
    
    with col2:
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        last_user_msg = None
        last_bot_msg = None
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                # Only show if not a repeat of the last user message
                if message['content'] != last_user_msg:
                    st.markdown(f'<div style="background:#e3f2fd;padding:10px;border-radius:8px;margin-bottom:5px;text-align:right;">You: {message["content"]}</div>', unsafe_allow_html=True)
                last_user_msg = message['content']
            else:
                # Only show if not a repeat of the last bot message
                if message['content'] != last_bot_msg:
                    st.markdown(f'<div style="background:#f3e5f5;padding:10px;border-radius:8px;margin-bottom:10px;text-align:left;">Bot: {message["content"]}</div>', unsafe_allow_html=True)
                last_bot_msg = message['content']
        
        # Input area
        user_input = st.text_input(
            "",
            value=st.session_state.user_input,
            key="user_input_box",
            placeholder="Type your question or select from the left..."
        )
        
        # Process input
        if user_input:
            # Only add user message if not a repeat
            if not (st.session_state.chat_history and st.session_state.chat_history[-1]['type'] == 'user' and st.session_state.chat_history[-1]['content'] == user_input):
                st.session_state.chat_history.append({'type': 'user', 'content': user_input})
            response = chatbot.get_response(user_input)
            # Gentle fallback with suggestions
            fallback = ("I'm here to answer questions about our services. "
                        "Please select a question on the left or ask about one of these topics:\n- " +
                        "\n- ".join(FAQ_DATA['questions'][:3]) + "\n...and more!")
            # Only add fallback if not a repeat
            if response['answer'] is None:
                last_bot_msg = next((msg['content'] for msg in reversed(st.session_state.chat_history) if msg['type'] == 'bot'), None)
                if last_bot_msg != fallback:
                    st.session_state.chat_history.append({'type': 'bot', 'content': fallback})
            else:
                # Only add bot answer if not a repeat
                if not (st.session_state.chat_history and st.session_state.chat_history[-1]['type'] == 'bot' and st.session_state.chat_history[-1]['content'] == response['answer']):
                    st.session_state.chat_history.append({'type': 'bot', 'content': response['answer']})
            st.session_state.user_input = ''
            st.rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.user_input = ''
            st.rerun()
        
        st.header("ℹ️ About")
        st.write("This FAQ chatbot helps answer common questions about our services.")
        st.write("**How it works:**")
        st.write("1. Select a question")
        st.write("2. Get instant answers")
        st.write("3. View confidence score")
        st.write("4. Access quick FAQs")

if __name__ == "__main__":
    main()
