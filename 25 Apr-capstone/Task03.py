# ================================
# Intelligent Customer Support Chatbot
# Azure AI Foundry Style Project (Final Fixed)
# ================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import random

app = FastAPI()

# ================================
# Load Datasets
# ================================

chat_data = pd.read_csv("customer_support_data.csv")   # Query, Intent, Response
sales_data = pd.read_csv("retail_sales_data.csv")      # Product data
sensor_data = pd.read_csv("machine_sensor_data.csv")   # Machine data

# ================================
# Request Model
# ================================

class ChatRequest(BaseModel):
    user_id: int
    query: str

# ================================
# Context Memory
# ================================

user_context = {}

# ================================
# NLP Intent Detection (Improved)
# ================================

def detect_intent(query):
    query = query.lower()

    if "recommend" in query:
        return "recommendation"
    elif "issue" in query or "problem" in query:
        return "ticket"
    elif "machine" in query or "sensor" in query:
        return "machine"
    else:
        return "chatbot"

# ================================
# Chatbot (Dataset-based NLP)
# ================================

def get_chat_response(query):
    query = query.lower()

    for _, row in chat_data.iterrows():
        dataset_query = row["Query"].lower()

        # keyword matching
        if any(word in query for word in dataset_query.split()):
            return row["Response"]

    return "Sorry, I didn't understand your request."

# ================================
# Recommendation System
# ================================

def recommend_product():
    top = sales_data.sort_values(by="UnitsSold", ascending=False).iloc[0]
    return f"Top selling product is {top['ProductID']} in {top['Category']} category."

# ================================
# Ticket System
# ================================

def create_ticket(issue):
    ticket_id = random.randint(1000, 9999)
    return f"Support ticket created with ID {ticket_id} for issue: {issue}"

# ================================
# Machine Monitoring
# ================================

def check_machine_health():
    high_temp = sensor_data[sensor_data["Temperature"] > 90]
    failures = sensor_data[sensor_data["Failure"] == 1]

    if not failures.empty:
        return "Critical Alert: Machine failure detected!"

    if not high_temp.empty:
        return "Warning: Some machines have high temperature."

    return "All machines are operating normally."

# ================================
# Chat API
# ================================

@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    query = request.query

    # Save context
    user_context[user_id] = query

    intent = detect_intent(query)

    # ============================
    # Intent Handling
    # ============================

    # 🔹 Recommendation
    if intent == "recommendation":
        return {"response": recommend_product()}

    # 🔹 Ticket system
    elif intent == "ticket":
        return {"response": create_ticket(query)}

    # 🔹 Machine monitoring
    elif intent == "machine":
        return {"response": check_machine_health()}

    # 🔹 Chatbot NLP
    else:
        return {"response": get_chat_response(query)}

# ================================
# Run Server
# ================================

# Run:
# uvicorn app:app --reload