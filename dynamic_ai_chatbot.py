import os
print("Current working directory:", os.getcwd())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


file_path = r'C:\AI_Chatbot_Project\datasets\Simple Dialogs for Chatbot\dialogs.txt'

# Read the file into a list of lines
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

dialog_lines = [line for line in lines if line.strip() != '']

# Preview the first 10 lines
for line in dialog_lines[:10]:
    print(line)


# In[3]:


# Pair consecutive lines as Q (prompt) and A (response)
qa_pairs = []
for i in range(0, len(dialog_lines) - 1, 2):
    qa_pairs.append((dialog_lines[i], dialog_lines[i+1]))

# Preview first 5 Q&A pairs
for q, a in qa_pairs[:5]:
    print("Q:", q)
    print("A:", a)
    print("---")


# In[4]:


def simple_retrieval_bot(user_input, qa_pairs):
    # Find the closest match using string similarity (basic example)
    from difflib import SequenceMatcher
    scores = [(SequenceMatcher(None, user_input.lower(), q.lower()).ratio(), a) for q, a in qa_pairs]
    scores.sort(reverse=True)
    # Return the response with highest similarity score
    return scores[0][1] if scores else "Sorry, I don't understand."

# Example usage
user_message = "how are you doing today?"
bot_response = simple_retrieval_bot(user_message, qa_pairs)
print("BOT:", bot_response)


# In[5]:


import pandas as pd

df = pd.read_csv(r'20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv')
print(df.head())
print(df.columns)


# In[6]:


# Define columns based on your DataFrame
q_col = 'utterance'   # column for user input
a_col = 'intent'      # column for chatbot intent

# Select utterance-intent pairs
qa_pairs = list(zip(df[q_col], df[a_col]))

# Preview first few pairs
for q, a in qa_pairs[:5]:
    print('User:', q)
    print('Intent:', a)
    print('---')


# In[7]:


from difflib import SequenceMatcher

def simple_retrieval_bot(user_input, qa_pairs):
    scores = [(SequenceMatcher(None, user_input.lower(), q.lower()).ratio(), a) for q, a in qa_pairs]
    scores.sort(reverse=True)
    return scores[0][1] if scores else "Sorry, I don't understand."

# Example usage
user_message = "I want to check my order status"
bot_response = simple_retrieval_bot(user_message, qa_pairs)
print('BOT:', bot_response)


# In[8]:


df = pd.read_csv(r'Bitext_Sample_Customer_Service_Training_Dataset.csv')
print(df.head())
print(df.columns)


# In[9]:


# Pair consecutive lines as Q (prompt) and A (response)
qa_pairs = []
for i in range(0, len(dialog_lines) - 1, 2):
    qa_pairs.append((dialog_lines[i], dialog_lines[i+1]))

# Preview first 5 Q&A pairs
for q, a in qa_pairs[:5]:
    print("Q:", q)
    print("A:", a)
    print("---")


# In[10]:


from difflib import SequenceMatcher

def simple_retrieval_bot(user_input, qa_pairs):
    scores = [(SequenceMatcher(None, user_input.lower(), q.lower()).ratio(), a) for q, a in qa_pairs]
    scores.sort(reverse=True)
    return scores[0][1] if scores else "Sorry, I don't understand."

# Example usage
user_message = "I want to check my order status"
bot_response = simple_retrieval_bot(user_message, qa_pairs)
print('BOT:', bot_response)


# In[11]:


lines_path = r'C:\AI_Chatbot_Project\datasets\cornell_movie_dialogs_corpus\cornell movie-dialogs corpus\movie_lines.txt'

# Parse manually for complete robustness
data = []
with open(lines_path, encoding="ISO-8859-1") as f:
    for line in f:
        parts = line.strip().split("+++$+++")
        if len(parts) == 5:
            data.append([part.strip() for part in parts])

import pandas as pd
lines_df = pd.DataFrame(data, columns=['lineID', 'characterID', 'movieID', 'character', 'text'])
print(lines_df.head())


# In[12]:


conversations_path = r'C:\AI_Chatbot_Project\datasets\cornell_movie_dialogs_corpus\cornell movie-dialogs corpus\movie_conversations.txt'
conv_columns = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']
conv_df = pd.read_csv(conversations_path, sep=' \+\+\+\$\+\+\+ ', engine='python', names=conv_columns, encoding='utf-8')

print(conv_df.head())


# In[13]:


import ast

def build_qa_pairs(conv_df, lines_df, max_conversations=500):
    qa_pairs = []
    conv_sample = conv_df.head(max_conversations)  # Limit to first N conversations
    for i in range(len(conv_sample)):
        utterance_ids = ast.literal_eval(conv_sample.loc[i, 'utteranceIDs'])
        texts = [lines_df.loc[lines_df['lineID'] == uid, 'text'].values[0] for uid in utterance_ids]
        # Pair consecutive utterances
        for j in range(len(texts) - 1):
            qa_pairs.append((texts[j], texts[j + 1]))
    return qa_pairs

# Usage with sample size for faster processing
qa_pairs = build_qa_pairs(conv_df, lines_df, max_conversations=3000)

# Preview first 5 Q&A pairs
for q, a in qa_pairs[:5]:
    print('Q:', q)
    print('A:', a)
    print('---')


# # Data Preprocessing

# In[14]:


import re

def clean_text(text):
    text = text.lower()
    text = text.strip()
    # Replace common contractions
    contractions = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "can't": "cannot",
        "won't": "will not",
        "didn't": "did not",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not"
    }
    for c, full in contractions.items():
        text = re.sub(r'\b' + c + r'\b', full, text)
    # Remove all non-alphanumeric except basic punctuations . , ? !
    text = re.sub(r"[^a-z0-9.,?!' ]+", '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# In[15]:


# Clean all Q&A pairs
cleaned_qa_pairs = [(clean_text(q), clean_text(a)) for q, a in qa_pairs]

# Preview a few cleaned pairs
for q, a in cleaned_qa_pairs[:5]:
    print('Q:', q)
    print('A:', a)
    print('---')


# In[16]:


# Remove pairs with empty questions or answers
cleaned_qa_pairs = [(q, a) for q, a in cleaned_qa_pairs if q and a]

# Remove duplicate pairs
cleaned_qa_pairs = list(set(cleaned_qa_pairs))

print(f'Total pairs after cleaning: {len(cleaned_qa_pairs)}')


# In[17]:


empty_q = sum(1 for q, a in cleaned_qa_pairs if not q.strip())
empty_a = sum(1 for q, a in cleaned_qa_pairs if not a.strip())
print("Empty questions:", empty_q)
print("Empty answers:", empty_a)


# In[18]:


total = len(cleaned_qa_pairs)
unique = len(set(cleaned_qa_pairs))
print(f"Total pairs: {total}, Unique pairs: {unique}")
print("Duplicates removed:", total - unique)


# In[19]:


print("Sample cleaned Q&A pairs:")
for q, a in cleaned_qa_pairs[:10]:
    print("Q:", q)
    print("A:", a)
    print("---")


# In[20]:


q_lengths = [len(q.split()) for q, a in cleaned_qa_pairs]
a_lengths = [len(a.split()) for q, a in cleaned_qa_pairs]

print(f"Average question length (words): {sum(q_lengths)/len(q_lengths):.2f}")
print(f"Average answer length (words): {sum(a_lengths)/len(a_lengths):.2f}")


# # Basic Retrieval Bot Code

# In[21]:


from difflib import SequenceMatcher

def simple_retrieval_bot(user_input, cleaned_qa_pairs):
    scores = [(SequenceMatcher(None, user_input.lower(), q.lower()).ratio(), a) for q, a in cleaned_qa_pairs]
    scores.sort(reverse=True)
    best_score, best_answer = scores[0]
    if best_score > 0.5:   # threshold to avoid random matches
        return best_answer
    else:
        return "Sorry, I don't understand."

# Test example:
user_message = "how are you doing today?"
bot_response = simple_retrieval_bot(user_message, cleaned_qa_pairs)
print("BOT:", bot_response)


# # Advanced NLP

# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
questions = [q for q, a in cleaned_qa_pairs]

# Vectorize questions using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def enhanced_retrieval_bot(user_input):
    user_vec = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_idx]
    if best_score > 0.3:  # threshold, can be tuned
        return cleaned_qa_pairs[best_idx][1]
    else:
        return "Sorry, I don't understand."

# Testing the enhanced bot
user_message = "How are you doing today?"
bot_response = enhanced_retrieval_bot(user_message)
print("BOT:", bot_response)


# In[23]:


print("Welcome to your chatbot! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = enhanced_retrieval_bot(user_input)
    print("Chatbot:", response)


# # Improving and Correcting Responses

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string

# Basic text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    return text

# Preprocess questions
questions = [preprocess_text(q) for q, a in cleaned_qa_pairs]
answers = [a for q, a in cleaned_qa_pairs]

# Vectorize with TF-IDF including bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(questions)

def enhanced_tfidf_bot(user_input):
    user_input_processed = preprocess_text(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_idx]
    
    if best_score > 0.35:  # tune threshold based on experiments
        return answers[best_idx]
    else:
        return "Sorry, I didn't quite get that. Can you please rephrase?"

# Example test
print(enhanced_tfidf_bot("How are you doing today?"))
print(enhanced_tfidf_bot("Tell me about your services."))


# In[25]:


# Sample labeled data: (question, intent)
labeled_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("how are you", "greeting"),
    ("where is my order", "order_status"),
    ("track my package", "order_status"),
    ("can you help me", "help_request"),
    ("i need assistance", "help_request"),
    ("bye", "goodbye"),
    ("exit", "goodbye"),
]


# In[26]:


import pandas as pd

df_intents = pd.DataFrame(labeled_data, columns=["question", "intent"])


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Create a pipeline: TF-IDF vectorizer + Logistic Regression classifier
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words="english"), LogisticRegression())

# Train model on labeled intent data
model.fit(df_intents["question"], df_intents["intent"])


# In[28]:


intent_responses = {
    "greeting": "Hello! How can I help you today?",
    "order_status": "Please provide your order ID, and I will check the status for you.",
    "help_request": "Sure! What do you need help with?",
    "goodbye": "Goodbye! Have a nice day!",
    "unknown": "Sorry, I didn't understand that. Could you please rephrase?",
}


# In[29]:


def intent_chatbot(user_input):
    intent = model.predict([user_input])[0]
    return intent_responses.get(intent, intent_responses["unknown"])


# In[30]:


print(intent_chatbot("Hi there"))
print(intent_chatbot("Can you check my order?"))
print(intent_chatbot("Thanks, bye"))
print(intent_chatbot("What time is it?"))  # Will likely trigger unknown


# In[31]:


# Initialize an empty list to store conversation history
chat_history = []

# Max history length to keep
max_history_length = 5

def add_to_history(user_input, bot_response):
    # Add the latest exchange as a tuple
    chat_history.append((user_input, bot_response))
    # Trim history if longer than max allowed
    if len(chat_history) > max_history_length:
        chat_history.pop(0)

# Sample usage in a chatbot loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    # Here you would call your response function; for demo, we just echo
    bot_response = "This is a placeholder reply."
    
    # Print and save the latest exchange
    print("Chatbot:", bot_response)
    add_to_history(user_input, bot_response)

    # Optional: print current history (for debugging)
    # print("Current chat history:", chat_history)


# In[32]:


def get_combined_input(user_input, chat_history, max_context=2):
    # Get last few user inputs from history (excluding current)
    recent_inputs = [turn[0] for turn in chat_history[-max_context:]]
    # Combine them with current input
    combined = " ".join(recent_inputs + [user_input])
    return combined

def context_aware_response(user_input, chat_history):
    combined_input = get_combined_input(user_input, chat_history)
    # Use combined_input for similarity-based retrieval or intent recognition
    # Example using enhanced_tfidf_bot from previous steps
    response = enhanced_tfidf_bot(combined_input)
    return response

# Example loop usage:

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    bot_response = context_aware_response(user_input, chat_history)
    
    print("Chatbot:", bot_response)
    
    chat_history.append((user_input, bot_response))


# In[33]:


# Example: Simple rule-based intent recognizer for basic conversation
def rule_based_intent(user_input):
    user_input = user_input.lower()
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening']
    byes = ['bye', 'goodbye', 'see you']
    how_are_you = ['how are you', 'how are you doing']
    
    if any(phrase in user_input for phrase in greetings):
        return "greeting"
    elif any(phrase in user_input for phrase in byes):
        return "goodbye"
    elif any(phrase in user_input for phrase in how_are_you):
        return "how_are_you"
    else:
        return None

# Direct intent-to-response mapping
intent_responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a nice day!",
    "how_are_you": "I'm just a bot, but I'm here to help you!"
}

def smart_context_aware_response(user_input, chat_history):
    intent = rule_based_intent(user_input)
    if intent and intent in intent_responses:
        return intent_responses[intent]
    else:
        combined_input = get_combined_input(user_input, chat_history)
        response = enhanced_tfidf_bot(combined_input)
        return response

# Same interactive loop, just use smart response
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    bot_response = smart_context_aware_response(user_input, chat_history)
    print("Chatbot:", bot_response)
    chat_history.append((user_input, bot_response))


# ***Sentiment Analysis***

# In[34]:


# get_ipython().system('python -m textblob.download_corpora')


# In[35]:


def detect_sentiment(user_input):
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# Example usage in your smart response function
def smart_context_aware_response(user_input, chat_history):
    intent = rule_based_intent(user_input)
    sentiment = detect_sentiment(user_input)
    
    if intent and intent in intent_responses:
        # Optionally tailor response based on sentiment
        if intent == "greeting" and sentiment == "negative":
            return "Hello! You sound a bit down. How can I assist you?"
        return intent_responses[intent]
    else:
        combined_input = get_combined_input(user_input, chat_history)
        response = enhanced_tfidf_bot(combined_input)
        # Optionally adjust fallback response
        if sentiment == "negative":
            response = "I'm sorry if something's bothering you. " + response
        return response


# In[36]:


from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def detect_sentiment(user_input):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(user_input)['compound']
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"


# In[37]:


from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# (You only need to download the lexicon once)
# nltk.download('vader_lexicon')

def detect_sentiment(user_input):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(user_input)['compound']
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"


# In[38]:


def rule_based_intent(user_input):
    user_input = user_input.lower()
    # Existing intents...

    negative_phrases = ["i hate you", "you are bad", "you suck", "i am angry"]
    
    # Check sensitive negative expressions first
    if any(phrase in user_input for phrase in negative_phrases):
        return "negative_emotion"

    # Existing checks...
    
    # If no intent matched
    return None

# Extend intent responses
intent_responses.update({
    "negative_emotion": "I'm sorry to hear that. I'm here to help if you want to talk."
})


# In[39]:


# get_ipython().system('python -m spacy download en_core_web_sm')


# In[40]:


import sys
# get_ipython().system('"{sys.executable}" -m pip install spacy')


# In[41]:


# get_ipython().system('"{sys.executable}" -m spacy download en_core_web_sm')


# In[42]:


import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage in chatbot flow:
user_input = "I want to order a pizza for tomorrow"
entities = extract_entities(user_input)
print("Entities:", entities)


# In[43]:


def rule_based_intent(user_input):
    user_input = user_input.lower()
    
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening']
    byes = ['bye', 'goodbye', 'see you']
    how_are_you = ['how are you', 'how are you doing']
    thanks = ['thank you', 'thanks', 'thx']
    negative_phrases = ['i hate you', 'you are bad', 'you suck', 'i am angry', 'you annoy me']
    
    if any(phrase in user_input for phrase in negative_phrases):
        return "negative_emotion"
    elif any(phrase in user_input for phrase in greetings):
        return "greeting"
    elif any(phrase in user_input for phrase in byes):
        return "goodbye"
    elif any(phrase in user_input for phrase in how_are_you):
        return "how_are_you"
    elif any(phrase in user_input for phrase in thanks):
        return "thanks"
    else:
        return None

# Extend intent responses dictionary
intent_responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a nice day!",
    "how_are_you": "I'm just a bot, but I'm here to help you!",
    "thanks": "You're welcome!",
    "negative_emotion": "I'm sorry to hear that. I'm here if you want to talk or need assistance."
}


# In[44]:


def smart_context_aware_response(user_input, chat_history):
    # Step 1: Intent recognition
    intent = rule_based_intent(user_input)
    
    # Step 2: Sentiment analysis
    sentiment = detect_sentiment(user_input)
    
    # Step 3: Extract named entities
    entities = extract_entities(user_input)
    
    # Step 4: Generate response based on intent
    if intent and intent in intent_responses:
        # Tailor response for negative greeting
        if intent == "greeting" and sentiment == "negative":
            return "Hello! You sound a bit down. How can I assist you?"
        # Tailor for negative emotion
        if intent == "negative_emotion":
            return intent_responses[intent]
        
        # General intent response
        return intent_responses[intent]
    
    # Step 5: Contextual retrieval fallback
    combined_input = get_combined_input(user_input, chat_history)
    response = enhanced_tfidf_bot(combined_input)
    
    # Step 6: Optionally customize response for negative sentiment
    if sentiment == "negative":
        response = "I'm sorry if something is bothering you. " + response
    
    # Step 7: Optionally mention extracted entities (for better UX)
    if entities:
        entity_list = ', '.join([f"{text} ({label})" for text, label in entities])
        response += f" By the way, I noticed you mentioned: {entity_list}."
    
    return response


# In[45]:


def rule_based_intent(user_input):
    user_input = user_input.lower()
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening']
    byes = ['bye', 'goodbye', 'see you']
    how_are_you = ['how are you', 'how are you doing']
    thanks = ['thank you', 'thanks', 'thx']
    negative_phrases = ['i hate you', 'you are bad', 'you suck', 'i am angry', 'you annoy me']
    order_phrases = ['order', 'book', 'buy', 'reserve']

    if any(phrase in user_input for phrase in negative_phrases):
        return "negative_emotion"
    elif any(phrase in user_input for phrase in greetings):
        return "greeting"
    elif any(phrase in user_input for phrase in byes):
        return "goodbye"
    elif any(phrase in user_input for phrase in how_are_you):
        return "how_are_you"
    elif any(phrase in user_input for phrase in thanks):
        return "thanks"
    elif any(word in user_input for word in order_phrases):
        return "order"
    else:
        return None

intent_responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a nice day!",
    "how_are_you": "I'm just a bot, but I'm here to help you!",
    "thanks": "You're welcome!",
    "negative_emotion": "I'm sorry to hear that. I'm here if you want to talk or need assistance.",
    "order": "Sure! What would you like to order?"
}

def smart_context_aware_response(user_input, chat_history):
    # Step 1: Intent recognition
    intent = rule_based_intent(user_input)
    # Step 2: Sentiment analysis
    sentiment = detect_sentiment(user_input)
    # Step 3: Extract named entities
    entities = extract_entities(user_input)
    # Step 4: Generate response based on intent
    if intent and intent in intent_responses:
        if intent == "greeting" and sentiment == "negative":
            return "Hello! You sound a bit down. How can I assist you?"
        if intent == "negative_emotion":
            return intent_responses[intent]
        # Customizable behavior for "order" intent
        if intent == "order":
            if entities:
                entity_list = ', '.join([f"{text} ({label})" for text, label in entities])
                return f"Okay, I can help place your order for: {entity_list}."
            else:
                return "Sure! What would you like to order?"
        return intent_responses[intent]
    # Fallback: retrieval + sentiment + entities
    combined_input = get_combined_input(user_input, chat_history)
    response = enhanced_tfidf_bot(combined_input)
    if sentiment == "negative":
        response = "I'm sorry if something is bothering you. " + response
    if entities:
        entity_list = ', '.join([f"{text} ({label})" for text, label in entities])
        response += f" By the way, I noticed you mentioned: {entity_list}."
    return response


# In[46]:


# Example usage
chat_history = []
response = smart_context_aware_response("Can I order a pizza for tomorrow?", chat_history)
print(response)


# In[47]:


dialogue_state = {
    "current_state": "START",  # Conversation state
    "slots": {
        "item": None,
        "delivery_date": None
    }
}

def multi_turn_order_bot(user_input, dialogue_state):
    user_input_lower = user_input.lower()
    state = dialogue_state["current_state"]
    slots = dialogue_state["slots"]

    # Extract entities
    entities = extract_entities(user_input)
    entity_dict = {label: text for text, label in entities}   # <-- fixed

    if state == "START":
        dialogue_state["current_state"] = "ASK_ITEM"
        return "Welcome! What would you like to order?"

    elif state == "ASK_ITEM":
        if "item" in entity_dict:
            slots["item"] = entity_dict["item"]
            dialogue_state["current_state"] = "ASK_DATE"
            return f"Got it, you want to order {slots['item']}. When should I deliver it?"
        else:
            slots["item"] = user_input.strip()
            dialogue_state["current_state"] = "ASK_DATE"
            return f"Okay, ordering {slots['item']}. What delivery date do you want?"

    elif state == "ASK_DATE":
        if "DATE" in entity_dict:
            slots["delivery_date"] = entity_dict["DATE"]
            dialogue_state["current_state"] = "CONFIRM"
            return f"You want {slots['item']} delivered on {slots['delivery_date']}. Is that correct? (yes/no)"
        else:
            slots["delivery_date"] = user_input.strip()
            dialogue_state["current_state"] = "CONFIRM"
            return f"Ok, delivery date set as {slots['delivery_date']}. Is that correct? (yes/no)"

    elif state == "CONFIRM":
        if user_input_lower in ["yes", "y"]:
            dialogue_state["current_state"] = "END"
            return f"Thanks! Your order for {slots['item']} on {slots['delivery_date']} is confirmed."
        elif user_input_lower in ["no", "n"]:
            dialogue_state["current_state"] = "ASK_ITEM"
            slots["item"] = None
            slots["delivery_date"] = None
            return "Okay, let's start over. What would you like to order?"
        else:
            return "Please answer 'yes' or 'no'. Is your order correct?"

    elif state == "END":
        return "Your order is already confirmed. If you'd like to order something else, please start a new session."

    return "Sorry, I didn't understand that. Let's start over. What would you like to order?"


# In[48]:


def unified_chatbot(user_input, dialogue_state, chat_history):
    """
    Decide whether to handle the input as an order (multi-turn)
    or a general query (intent/sentiment/NER).
    """
    # If we are inside an active order flow (not END)
    if dialogue_state["current_state"] != "END":
        return multi_turn_order_bot(user_input, dialogue_state)

    # Otherwise, treat as general conversation
    return smart_context_aware_response(user_input, chat_history)


# Example interactive loop for the unified bot
dialogue_state = {
    "current_state": "START",
    "slots": {"item": None, "delivery_date": None},
}
chat_history = []

print("Welcome to the Dynamic AI Chatbot! Type 'exit' to quit.")
while True:
    user_in = input("You: ")
    if user_in.lower() == "exit":
        print("Bot: Goodbye!")
        break

    bot_out = unified_chatbot(user_in, dialogue_state, chat_history)
    print("Bot:", bot_out)

    # Keep track of context for retrieval/sentiment
    chat_history.append((user_in, bot_out))


# In[49]:


def smart_context_aware_response(user_input, chat_history):
    user_input_lower = user_input.strip().lower()

    # --- Simple rules for common casual replies ---
    if user_input_lower in ["ok", "okay", "alright", "fine", "cool", "sure"]:
        return "Got it! Let me know if you need anything else."

    if user_input_lower in ["thanks", "thank you", "thx"]:
        return "You're welcome! Happy to help. 😊"

    # --- Intent recognition ---
    intent = rule_based_intent(user_input)

    # --- Sentiment analysis ---
    sentiment = detect_sentiment(user_input)

    # --- Named Entity Recognition ---
    entities = extract_entities(user_input)

    # --- Intent-based responses ---
    if intent and intent in intent_responses:
        if intent == "greeting" and sentiment == "negative":
            return "Hello! You sound a bit down. How can I assist you?"
        if intent == "negative_emotion":
            return intent_responses[intent]
        return intent_responses[intent]

    # --- Retrieval fallback ---
    combined_input = get_combined_input(user_input, chat_history)
    response = enhanced_tfidf_bot(combined_input)

    # Adjust for negative sentiment
    if sentiment == "negative":
        response = "I'm sorry if something is bothering you. " + response

    # Mention entities if found
    if entities:
        entity_list = ', '.join([f"{text} ({label})" for text, label in entities])
        response += f" By the way, I noticed you mentioned: {entity_list}."

    return response


# In[50]:


# ==========================================
# 54. Multi-turn Order Bot (auto-end after confirmation)
# ==========================================

def multi_turn_order_bot(user_input, dialogue_state):
    """
    Handles a simple multi-turn conversation for placing an order.
    Fills two slots: 'item' and 'delivery_date', with confirmation.
    Ends politely after order confirmation.
    """
    user_input_lower = user_input.lower()
    state = dialogue_state["current_state"]
    slots = dialogue_state["slots"]

    # --- Extract entities ---
    entities = extract_entities(user_input)            # -> [(text, label), ...]
    entity_dict = {label: text for text, label in entities}

    # --- State machine logic ---
    if state == "START":
        dialogue_state["current_state"] = "ASK_ITEM"
        return "Welcome! What would you like to order?"

    elif state == "ASK_ITEM":
        if "item" in entity_dict:
            slots["item"] = entity_dict["item"]
            dialogue_state["current_state"] = "ASK_DATE"
            return f"Got it, you want to order {slots['item']}. When should I deliver it?"
        else:
            slots["item"] = user_input.strip()
            dialogue_state["current_state"] = "ASK_DATE"
            return f"Okay, ordering {slots['item']}. What delivery date do you want?"

    elif state == "ASK_DATE":
        if "DATE" in entity_dict:
            slots["delivery_date"] = entity_dict["DATE"]
            dialogue_state["current_state"] = "CONFIRM"
            return f"You want {slots['item']} delivered on {slots['delivery_date']}. Is that correct? (yes/no)"
        else:
            slots["delivery_date"] = user_input.strip()
            dialogue_state["current_state"] = "CONFIRM"
            return f"Ok, delivery date set as {slots['delivery_date']}. Is that correct? (yes/no)"

    elif state == "CONFIRM":
        if user_input_lower in ["yes", "y"]:
            dialogue_state["current_state"] = "END"
            return (f"Thanks! Your order for {slots['item']} on {slots['delivery_date']} is confirmed. "
                    f"Goodbye, and have a great day! 👋")
        elif user_input_lower in ["no", "n"]:
            dialogue_state["current_state"] = "ASK_ITEM"
            slots["item"] = None
            slots["delivery_date"] = None
            return "Okay, let's start over. What would you like to order?"
        else:
            return "Please answer 'yes' or 'no'. Is your order correct?"

    elif state == "END":
        return "Your order is already confirmed. If you'd like to order something else, please restart the session."

    return "Sorry, I didn't understand that. Let's start over. What would you like to order?"


# In[51]:


# ==========================================
# 55. Unified Chatbot (auto-exit after order confirmation)
# ==========================================

def unified_chatbot(user_input, dialogue_state, chat_history):
    """
    Unified chatbot that decides whether to handle order flow
    or general queries. Ends the conversation automatically
    after order confirmation.
    """
    # If we are still inside order flow (not END)
    if dialogue_state["current_state"] != "END":
        return multi_turn_order_bot(user_input, dialogue_state)

    # Once END is reached, exit gracefully
    return "SESSION_END"


# Example interactive loop for the unified bot
dialogue_state = {
    "current_state": "START",
    "slots": {"item": None, "delivery_date": None},
}
chat_history = []

print("Welcome to the Dynamic AI Chatbot! Type 'exit' to quit.")
while True:
    user_in = input("You: ")
    if user_in.lower() == "exit":
        print("Bot: Goodbye!")
        break

    bot_out = unified_chatbot(user_in, dialogue_state, chat_history)

    if bot_out == "SESSION_END":
        # Auto exit after order confirmation
        print("Bot: Your order is complete. Goodbye! 👋")
        break

    print("Bot:", bot_out)
    chat_history.append((user_in, bot_out))


# # Expanded GPT Response

# In[52]:


intent_responses = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a nice day!",
    "how_are_you": "I'm just a bot, but I'm here to help you!",
    "thanks": "You're welcome! Happy to help.",
    "negative_emotion": "I'm sorry to hear that. I'm here to help if you want to talk.",
    "order": "Sure! What would you like to order?",
    "confirmation_yes": "Got it! Let me know if you need anything else.",
    "confirmation_no": "Okay, let's start over. What would you like to order?",
    "casual_ok": "Alright, cool!",
    "casual_fine": "Fine, I'm glad you're okay.",
    "casual_sure": "Sure! What do you need?"
}


# In[53]:


def smart_context_aware_response(user_input, chat_history):
    intent = rule_based_intent(user_input)
    sentiment = detect_sentiment(user_input)
    entities = extract_entities(user_input)
    
    # Adjust reply for negative sentiment
    if intent == "greeting" and sentiment == "negative":
        return "Hello! You sound a bit down. How can I assist you?"
    
    if intent in intent_responses:
        if intent == "negative_emotion":
            return intent_responses[intent]
        elif intent == "order":
            entity_list = ", ".join([text for text, label in entities])
            if entity_list:
                return f"Okay, I can help place your order for {entity_list}."
            else:
                return intent_responses[intent]
        elif intent == "thanks":
            return intent_responses[intent]
        else:
            return intent_responses[intent]
    
    # If no matching intent, use context and fallback
    combined_input = get_combined_input(user_input, chat_history)
    response = enhanced_tfidf_bot(combined_input)
    if sentiment == "negative":
        response += " I'm sorry if something is bothering you."
    # Add entity extraction marker if entities are present
    if entities:
        entity_list = ", ".join([text for text, label in entities])
        response += f" By the way, I noticed you mentioned {entity_list}."
    return response


# In[54]:


chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    bot_response = smart_context_aware_response(user_input, chat_history)
    print("Chatbot:", bot_response)
    chat_history.append((user_input, bot_response))


# In[55]:


import random

intent_responses = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! 😊 What brings you here?",
        "Hey! How can I assist you?"
    ],
    "goodbye": [
        "Goodbye! Have a wonderful day!",
        "Thanks for chatting. Take care!",
        "See you later! 👋"
    ],
    "how_are_you": [
        "I'm just a bot, but I'm always ready to help you!",
        "I'm here, ready to assist. How can I help?",
        "All systems go! What can I do for you today?"
    ],
    "thanks": [
        "You're welcome! Anything else I can help with?",
        "My pleasure. Let me know if you need more help.",
        "Glad to be of service!"
    ],
    "negative_emotion": [
        "I'm sorry to hear that. If you need to talk, I'm here for you.",
        "That doesn't sound great—want to tell me more?",
        "Oh no, is there anything I can do to help?"
    ],
    "order": [
        "Great! What would you like to order today?",
        "Absolutely—just let me know what you want to order.",
        "Sure thing! What can I place for you?"
    ],
    "confirmation_yes": [
        "Awesome, your request is confirmed! 👍",
        "Done! If you have more requests, let me know.",
        "Got it! All set. 😊"
    ],
    "confirmation_no": [
        "No problem, let's try that again. What would you like to order?",
        "Alright, let's start over. What's on your mind?",
        "Okay, nothing's confirmed yet—please tell me what you want."
    ],
    "casual_ok": [
        "Alright, cool! 😊",
        "Okay, moving ahead!",
        "Great, let's keep going."
    ],
    "casual_fine": [
        "Glad to hear you're good!",
        "Happy to know all is well.",
        "That's great!"
    ],
    "casual_sure": [
        "Absolutely. What do you need?",
        "Of course! Ask away.",
        "You got it!"
    ]
}


# In[56]:


def pick_response(intent):
    """Randomly select a response from the template list for the detected intent."""
    if intent in intent_responses:
        return random.choice(intent_responses[intent])
    else:
        return None

def smart_context_aware_response(user_input, chat_history):
    intent = rule_based_intent(user_input)
    sentiment = detect_sentiment(user_input)
    entities = extract_entities(user_input)
    
    # Empathy and acknowledgment
    if intent == "greeting" and sentiment == "negative":
        return "Hello! You sound a bit down. How can I support you today?"
    
    if intent:
        reply = pick_response(intent)
        # Add acknowledgment for actions/requests
        if intent == "order":
            entity_list = ", ".join([text for text, label in entities])
            if entity_list:
                replies = [
                    f"Great! Placing your order for {entity_list}. When do you want it delivered?",
                    f"Awesome, I got {entity_list}! Tell me your preferred delivery time.",
                    f"Yum! You chose {entity_list}. When should I schedule the delivery?"
                ]
                reply = random.choice(replies)
            # Friendly nudge if entity not detected
            elif not entities:
                reply += " (Just let me know what you'd like to order!)"
        return reply
    
    # No intent: Use context/retrieval fallback
    combined_input = get_combined_input(user_input, chat_history)
    fallback = enhanced_tfidf_bot(combined_input)
    if not fallback or fallback.strip() == "":
        fallback = "Sorry, I didn't catch that. Could you please rephrase?"
    elif sentiment == "negative":
        fallback += " I'm really here to help if you're having a tough time."
    # If entities found, mention them
    if entities:
        entity_list = ", ".join([text for text, label in entities])
        fallback += f" (By the way, I noticed you mentioned {entity_list}.)"
    return fallback


# In[57]:


chat_history = []
print("Welcome to the improved Dynamic AI Chatbot! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    bot_response = smart_context_aware_response(user_input, chat_history)
    print("Chatbot:", bot_response)
    chat_history.append((user_input, bot_response))


# # Using LLMs



# In[61]:



import requests

import streamlit as st
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

def clean_citations(text):
    # Remove numeric citation markers like [1], [2], [3,4], etc.
    return re.sub(r'\[\d+(,\s*\d+)*\]', '', text).strip()

def perplexity_fallback(prompt, history=None):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if history and len(history) > 0:
        for turn in history[-4:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_turn, bot_turn = turn
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": bot_turn})

    messages.append({"role": "user", "content": prompt})

    json_data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 3000,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=json_data,
            timeout=60
        )
        # Defensive JSON parsing
        try:
            data = response.json()
        except Exception as e:
            print("Failed to decode JSON:", e)
            return "Sorry, the API response was not valid JSON.", []

        # Check data type before accessing keys
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                raw_answer = data["choices"][0].get("message", {}).get("content", "").strip()
                clean_answer = clean_citations(raw_answer)

                citations = data.get("citations", [])
                references = []
                if isinstance(citations, list):
                    for c in citations:
                        if isinstance(c, dict):
                            title = c.get("title", "Reference")
                            url = c.get("url", "")
                            if url:
                                references.append(f"{title}: {url}")

                return clean_answer, references

            else:
                print("API JSON missing choices or choices empty.")
                return "Sorry, no usable response from the API.", []

        else:
            print(f"API response JSON is not a dict: {type(data)}")
            return "Sorry, bad API response format.", []

    except Exception as e:
        print("Perplexity API Exception:", e)
        return "Sorry, there was an error generating a response.", []




# In[62]:


def smart_context_aware_response(user_input, chat_history):
    intent = rule_based_intent(user_input)
    sentiment = detect_sentiment(user_input)
    entities = extract_entities(user_input)

    if intent:
        reply = pick_response(intent)
        if intent == "order":
            entity_list = ", ".join([text for text, label in entities])
            if entity_list:
                replies = [
                    f"Great! Placing your order for {entity_list}. When do you want it delivered?",
                    f"Awesome, I got {entity_list}! Tell me your preferred delivery time.",
                    f"Yum! You chose {entity_list}. When should I schedule the delivery?"
                ]
                reply = random.choice(replies)
            elif not entities:
                reply += " (Just let me know what you'd like to order!)"
        return reply

    # Fallback to Perplexity API
    response, references = perplexity_fallback(user_input, chat_history)

    if sentiment == "negative":
        response += " If you're upset, I'm here to listen."

    if references:
        response += "\n\nReferences:\n" + "\n".join("- " + r for r in references)

    return response


# In[63]:


def enhanced_tfidf_bot(user_input):
    user_input_processed = preprocess_text(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_idx]
    
    if best_score > 0.35:  # your existing threshold
        return answers[best_idx]
    else:
        # Friendly fallback response (minute change)
        return "I'm not quite sure about that. Could you please rephrase or ask something else?"


# In[64]:


import time
from datetime import datetime

log_file = "chat_log.txt"
interaction_count = 0

chat_history = []
print("Welcome to the advanced Dynamic AI Chatbot powered by Perplexity! Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    if not user_input:  # Handle empty input gracefully
        print("Chatbot: Please say something so I can assist you.")
        continue

    start_time = time.time()
    bot_response = smart_context_aware_response(user_input, chat_history)
    end_time = time.time()

    print("Chatbot:", bot_response)
    print(f"[Response time: {end_time - start_time:.2f} seconds]")
    
    # Log conversation with timestamps
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - User: {user_input}\n")
        f.write(f"{timestamp} - Bot: {bot_response}\n")

    chat_history.append((user_input, bot_response))
    interaction_count += 1


# # Making a Streamlit App

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




