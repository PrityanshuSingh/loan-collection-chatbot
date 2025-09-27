# Loan Collection Chatbot

## Overview

The Loan Collection Chatbot is an intelligent, empathetic, and compliant solution designed to streamline loan recovery. It combines predictive modeling, persona-based strategies, and advanced conversational AI to deliver personalized, effective interactions. By leveraging a customer’s past financial history, behavioral data, and real-time intent signals, it adapts to each individual’s needs.
Built with **Streamlit** and **Python**, it is designed to be run easily and reliably.

---

## Features

-   **Predictive Default Model**: An XGBoost model that accurately predicts loan default probabilities.
-   **Persona Identification**: Uses KMeans clustering to segment customers into actionable personas (e.g., "Willing but Struggling," "High-Risk Avoider").
-   **Dynamic Conversational AI**: Powered by the Groq LLM to generate adaptive, human-like, and context-aware responses.
-   **RAG-Powered Compliance**: A Retrieval-Augmented Generation (RAG) system ensures all conversations adhere to internal policies and regulatory guidelines from a knowledge base.
-   **Real-time Analytics**: The UI displays live metrics on the conversation's health, including:
    -   Customer Sentiment & Intent
    -   Negative Interaction Strike Count
    -   Dynamic LLM-generated Strategy for each turn
    -   A Cumulative Success Score to grade the interaction.
-   **Containerized & Portable**: Packaged with a `Dockerfile` and `docker-compose.yml` for simple, one-command deployment.

---

## Getting Started with Docker (Recommended Method)

Running this project with Docker is the simplest way to get started. It handles all dependencies and configurations for you.

### Prerequisites

-   **Git**: To clone the repository.
-   **Docker Desktop**: Make sure it is installed and running on your system.

### 1. Clone the Repository

Open your terminal and clone the project to your local machine.

```bash
git clone https://github.com/your-username/loan-collection-chatbot.git
cd loan-collection-chatbot
```

### 2. Configure Your API Key

The chatbot requires an API key from [Groq](https://groq.com/) to function.

-   In the project's root directory, find the `config.py` file.
-   Open it and replace the placeholder with your actual Groq API key:

```python
# config.py

# Replace "gsk_..." with your key
GROQ_API_KEY = "gsk_YourActualGroqApiKeyHere"

# ... other settings
```

### 3. Build and Run the Container

With Docker Desktop running, use Docker Compose to build the image and start the application with a single command.

```bash
docker-compose up --build
```

-   This command reads the `docker-compose.yml` file.
-   The `--build` flag tells Docker to build the image from the `Dockerfile` if it doesn't exist or if the files have changed.
-   The process may take a few minutes the first time as it downloads the base image and installs all Python packages.

### 4. Access the Application

Once the container is running, open your web browser and navigate to:

**[http://localhost:8501](http://localhost:8501)**

### 5. Stopping the Application

To stop the container, return to your terminal and press `CTRL + C`. To clean up and remove the container completely, run:

```bash
docker-compose down
```

---

## Project Structure

```
loan-collection-chatbot/
│
├── app.py                   # Main Streamlit application file
├── config.py                # Configuration for API keys and paths
│
├── src/                     # Source code for the chatbot logic
│   ├── chatbot_engine.py    # The core IntegratedChatbot class
│   ├── rag_handler.py       # Functions for RAG setup and vectorstores
│   └── strategy_playbook.py # Persona-based conversation strategies
│
├── scripts/                 # Standalone scripts for model training
│   ├── train_persona_model.py
│   └── train_predictive_model.py
│
├── data/                    # Dataset file
│   └── Analytics_loan_collection_dataset.csv
│
├── assets/                  # Knowledge base files for RAG
│   ├── rbi_guidelines.pdf
│   └── ...
│
├── models/                  # Saved model artifacts (.pkl files)
│
├── Dockerfile               # Instructions to build the Docker image
├── docker-compose.yml       # Orchestration for running the container
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Dummy Config.py

```bash
# config.py

# --- API Keys ---
# IMPORTANT: It is recommended to use Streamlit secrets for deployment.
# For local development, you can set your API key here.
GROQ_API_KEY = "YOUR_API_KEY"

# --- Model Configuration ---
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# --- File Paths ---
DATA_FILE_PATH = 'data/Analytics_loan_collection_dataset.csv'
RAG_PDF_PATHS = [
    'assets/rbi_guidelines.pdf',
    'assets/best-practices-in-collections-strategies.pdf'
]

# --- Asset Paths (for saving/loading trained models) ---
PREDICTIVE_MODEL_ASSETS_PATH = 'models/predictive_assets.pkl'
PERSONA_MODEL_ASSETS_PATH = 'models/persona_assets.pkl'

# --- Chatbot Settings ---
MAX_NEGATIVE_STRIKES = 3
```

## Chatbot Screenshot
![WhatsApp Image 2025-09-16 at 07 05 24_cb57dae0](https://github.com/user-attachments/assets/945e81c2-0f36-48b7-bde3-cfbd9faccf36)

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
