import hashlib, random, math
from datetime import datetime
import time  # Import time at the top

# Simulated vector database (list of dicts with embeddings and metadata)
vector_db = []

def vectorize_text(text, dim=64):
    """Generates a deterministic simulated embedding for a given text."""
    # Use the text hash as seed to generate a fixed pseudo-random vector
    h = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) 
    random.seed(h)
    # Fixed-dimension vector with float values in range [0,1)
    return [random.random() for _ in range(dim)]

def route_model(question):
    """Decides which LLM model to use (simple heuristic based on question length)."""
    words = question.split()
    if len(words) < 6:
        return "openai/gpt-3.5-turbo"   # cheaper/faster model
    else:
        return "openai/gpt-4"           # more advanced (expensive and powerful) model

def call_model(model, question):
    """Simulates a call to the selected model via OpenRouter and returns the response."""
    # In a real scenario, here you would call OpenRouter API, e.g.:
    # openrouter.complete(prompt=..., model=model)
    return f"[Simulated response from model {model} to the question: '{question}']"

def store_interaction(question, answer, model):
    """Stores question & answer in the vector database with its embedding."""
    embedding = vectorize_text(question)
    log_entry = {
        "question": question,
        "answer": answer,
        "model": model,
        "embedding": embedding,
        "timestamp": datetime.now()
    }
    vector_db.append(log_entry)

def cosine(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def search_similar(question, threshold=0.8):
    """Searches for similar question in the vector database via cosine similarity."""
    if not vector_db:
        return None
    query_emb = vectorize_text(question)
    best_match = None
    best_score = 0.0
    for entry in vector_db:
        score = cosine(query_emb, entry["embedding"])
        if score > best_score:
            best_score = score
            best_match = entry
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None

if __name__ == "__main__":
    questions = [
        "How do I update my registration data by changing the delivery address?",
        "What is the product delivery time?",
        "How can I cancel my subscription?",
        "Is there interest-free installment payment?",
        "Which models are available?",
        "How do I update my registration data by changing the delivery address?",
        "What payment methods do you accept?",
        "How to change the delivery address?",
        "How can I return a defective product?",
        "Which models are available?",
        "What are your business hours?",
        "How do I get a second copy of the invoice?",
        "What are the benefits of the rewards club?",
        "How can I cancel my subscription?",            # Repeated
        "Is there interest-free installment payment?",  # Repeated
        "What is the product delivery time?",           # Repeated
        "How do I update my registration data?",        # Repeated
        "Is the website safe for purchases?",
        "Do you deliver on Saturdays?",
        "How can I contact support?",
        "How do I get a second copy of the invoice?",   # Repeated
        "Do you ship internationally?",
        "What are the benefits of the rewards club?",   # Repeated
        "How do I check my balance?",
        "How do I issue an electronic invoice?",
        "What online courses do you offer?",
        "How can I schedule a technical visit?",
        "How can I change my plan?",
        "I forgot my password, how do I recover it?",
        "How do I file a complaint with the ombudsman?",
        "What are the customer service channels?",
        "How can I cancel my subscription?",            # Repeated
        "How do I update my registration data?",        # Repeated
        "Is there interest-free installment payment?",  # Repeated
        "What payment methods do you accept?",          # Repeated
        "How can I contact support?"                    # Repeated
    ]

    for idx, user_question in enumerate(questions, 1):
        print(f"\n[{idx}] User asks: '{user_question}'")
        # 1. Check if a similar question has already been answered
        match = search_similar(user_question)
        if match:
            print("→ Retrieved from vector database (FAQ/HISTORY):")
            print(f"✔ Answer retrieved from FAQ: '{match[0]['answer']}' [Confidence: {match[1]:.2f}]")
        else:
            print("→ No similar answer found in the FAQ. Routing to LLM model...")
            # 2. Route to appropriate model
            selected_model = route_model(user_question)
            print(f"→ Selected model: {selected_model}")
            # 3. Simulate calling the model via OpenRouter
            answer = call_model(selected_model, user_question)
            # 4. Return response to user
            print(f"✔ Answer generated by the model: {answer}")
            # 5. Store interaction in vector database
            store_interaction(user_question, answer, selected_model)
        time.sleep(5)  # Waits 5 seconds before processing the next question

