# Custom-AI-Cosmetics-Chatbot

This is a custom chatbot application for cosmetics products, built with Flask, featuring user authentication, product management, and an AI-powered chatbot using fine-tuned language models.

## Features

- **User Authentication**: Sign up, login, and logout functionality.
- **Admin Panel**: Manage products (add, edit, delete) with admin access.
- **Product Catalog**: View and shop cosmetics products.
- **AI Chatbot**: Answer questions about cosmetics using a combination of fine-tuned BART model and LLaMA3.
- **Vector Search**: Uses ChromaDB for semantic search and retrieval.

## Architecture

The application uses:
- **Flask** for the web framework.
- **SQLAlchemy** for database management (SQLite).
- **Sentence Transformers** for similarity matching.
- **Transformers** library with a fine-tuned BART model for Q&A on cosmetics.
- **LangChain** with Chroma vectorstore and Ollama LLaMA3 for general queries.
- **ChromaDB** for persistent vector storage.

## Setup Instructions

1. **Clone or Download the Project**:
   - Ensure the `Final_llm_files/LLMfinal` directory is in your workspace.

2. **Install Dependencies**:
   - Navigate to the `website` directory.
   - Create a virtual environment: `python -m venv .venv`
   - Activate it: `source .venv/bin/activate` (on macOS/Linux)
   - Install requirements: `pip install -r requirements.txt`

3. **Model and Data Setup**:
   - The fine-tuned BART model is located in `trained_model_fullqa/`.
   - The Chroma vectorstore is in `chroma_index/`.
   - The Q&A dataset is in `fullqa_trimmed.jsonl`.
   - Note: Paths in `app.py` are hardcoded. Update them if necessary to match your file system.

4. **Ollama Setup**:
   - Ensure Ollama is installed and the `llama3` model is available: `ollama pull llama3`

5. **Run the Application**:
   - From the `website` directory: `python app.py`
   - Access at `http://127.0.0.1:5000/`

## Usage

- **Home Page**: View products.
- **Sign Up/Login**: Create an account or log in.
- **Admin Access**: Log in as admin (email: admin, password: admin) to manage products.
- **Shop**: Browse products.
- **Chat**: Interact with the chatbot via the `/chat` endpoint (likely integrated in the UI).

## Chatbot Logic

- The chatbot checks similarity between user query and dataset questions.
- If similarity >= 0.5, uses the fine-tuned BART model for precise answers.
- Otherwise, uses LLaMA3 for general responses.

## File Structure

- `website/app.py`: Main Flask application.
- `website/templates/`: HTML templates.
- `website/static/`: Static files (CSS, JS, images).
- `trained_model_fullqa/`: Fine-tuned BART model.
- `chroma_index/`: ChromaDB vectorstore.
- `fullqa_trimmed.jsonl`: Cosmetics Q&A dataset.

## Requirements

See `website/requirements.txt` for Python dependencies.

## Notes

- Database: SQLite (`db.sqlite3`).
- Default admin: email `admin`, password `admin`.
- Ensure all paths in `app.py` are correct for your system.
