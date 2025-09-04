# AI Fashion Recommendation Chatbot

This project is an intelligent, conversational AI chatbot designed to provide personalized fashion recommendations. It understands user queries in natural language, analyzes preferences for style, color, and fit, and searches a product database to find the perfect items. The chatbot also features advanced capabilities like analyzing user-uploaded images to offer style advice.

## Features

* **Conversational AI:** Powered by OpenAI's GPT models to provide friendly and context-aware fashion advice.
* **Advanced Keyword Analysis:** A sophisticated system to extract and understand fashion-specific terms (e.g., clothing types, sleeve lengths, fits, colors, styles) from user input.
* **Hybrid Product Search:** Combines traditional keyword matching (TF-IDF) with modern semantic search (Sentence Transformers) to find the most relevant products.
* **Multi-Item Requests:** Can handle queries for multiple clothing items at once (e.g., "a white shirt and blue jeans").
* **Contextual Filtering:** Automatically detects and applies filters for budget and gender based on the conversation.
* **Multi-Language Support:** Capable of translating user input and responses to support different languages.

## 🛠Technology Stack

* **Backend:** Python, FastAPI
* **AI & Machine Learning:**
    * OpenAI API (GPT-3.5-Turbo for chat, GPT-4o for vision)
    * Sentence Transformers (for semantic search)
    * Scikit-learn (for TF-IDF)
    * SpaCy (for language processing)
* **Database:**
    * SQLAlchemy ORM
    * PostgreSQL & MySQL
* **Frontend:** HTML, CSS, JavaScript (served with Jinja2 Templates)

## Getting Started (Local Setup)

Follow these instructions to get a copy of the project running on your local machine for development and testing.

### Prerequisites

* Python 3.8+
* Pip (Python package installer)
* A local MySQL or PostgreSQL database instance.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Nadiaaa11/Skripsiiii.git](https://github.com/Nadiaaa11/Skripsiiii.git)
    cd Skripsiiii
    ```

2.  **Create a virtual environment and activate it:**
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required libraries:**
    *Make sure you have created a `requirements.txt` file first by running `pip freeze > requirements.txt` in your local environment.*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory and add your secret keys and database URL. Your application will need these to run.
    ```
    # Example for a local MySQL database
    DATABASE_URL="mysql+aiomysql://root:your_password@localhost:3306/ecommerce"

    # Your OpenAI API Key
    OPENAI_API_KEY="sk-..."
    ```
    *Your Python code will need to be configured to load these variables.*

5.  **Run the application:**
    ```bash
    uvicorn Chatbot.main:app --reload
    ```
    The application will be running at `http://127.0.0.1:8000`.
