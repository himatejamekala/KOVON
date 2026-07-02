# Context-Aware Customer Support RAG Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support
questions from a company FAQ document, personalized using user details stored
in a SQLite database. Built with Python, LangChain, Groq API (`llama3-8b-8192`),
Chroma as the local vector store, and free local HuggingFace embeddings.

## Tech Stack

- Python 3.10+
- LLM: Groq API (`llama3-8b-8192`)
- Framework: LangChain
- Database: SQLite
- Vector Store: Chroma
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free)

## Project Structure

```
.
├── app.py              # Main chatbot (terminal loop + single-query CLI mode)
├── ingest.py            # Chunks company_faq.txt and builds the Chroma vector store
├── create_db.py          # Creates and seeds users.db (SQLite)
├── company_faq.txt        # Sample company FAQ knowledge base
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment file
└── README.md
```

## Setup

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API key**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set your real key:

   ```
   GROQ_API_KEY=your_actual_groq_api_key
   ```

4. **Create and seed the SQLite database**

   ```bash
   python create_db.py
   ```

   This creates `users.db` with a `users` table containing:

   | user_id | name         | membership_tier |
   |---------|--------------|------------------|
   | 101     | Riya Sharma  | Gold             |
   | 102     | Aman Verma   | Silver           |
   | 103     | Neha Iyer    | Platinum         |

5. **Build the vector store**

   ```bash
   python ingest.py
   ```

   This reads `company_faq.txt`, splits it into overlapping chunks, generates
   local embeddings, and persists them to `./chroma_db`.

## Running the Bot

### Interactive terminal mode

```bash
python app.py
```

You'll be prompted for a `user_id` and a question in a loop. Type `exit` to quit.

### Single-query mode (useful for quick testing)

```bash
python app.py --user_id 101 --query "What is the refund policy?"
```

## Sample Queries to Try

```bash
python app.py --user_id 101 --query "What is the refund policy?"
python app.py --user_id 103 --query "Do I get premium customer support?"
python app.py --user_id 999 --query "What are my benefits?"
python app.py --user_id 102 --query "Can I cancel my account?"
```

Expected behavior:

- **101, refund policy** → Answer grounded in the refund section, referencing
  Riya Sharma / Gold tier where relevant.
- **103, premium support** → Answer grounded in the premium support section,
  referencing Neha Iyer / Platinum tier.
- **999, any question** → `User not found. Please enter a valid user_id.`
- **102, cancellation** → Answer grounded only in the account cancellation
  section of the FAQ.

## Error Handling

- **Invalid `user_id`** → `User not found. Please enter a valid user_id.`
- **No relevant context retrieved** → `I do not have enough information in the
  provided knowledge base to answer this.`
- **Missing `GROQ_API_KEY`** → Clear message asking the user to set the
  environment variable.
- **Groq API errors / rate limits** → Caught and surfaced as a readable
  message instead of crashing the app.

## Notes

- Embeddings are generated locally and for free using
  `sentence-transformers/all-MiniLM-L6-v2`, so no embedding API key is
  required — only `GROQ_API_KEY` for the LLM call.
- The prompt strictly instructs the model to answer only from retrieved
  context and to say it doesn't know when the context is insufficient, to
  minimize hallucination.
- Re-run `python ingest.py` any time `company_faq.txt` changes, to refresh
  the vector store.
