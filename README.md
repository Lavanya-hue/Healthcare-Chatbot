# Healthcare Chatbot (NER + Classification + RAG)

This repo implements a healthcare information chatbot that demonstrates:
- Named Entity Recognition (NER)
- Text Classification (zero-shot)
- Retrieval-Augmented Generation (FAISS + sentence-transformers)
- Local generation with `google/flan-t5-base`
- FastAPI server + Docker container

## Quick setup (local)
1. Clone repo
2. Create and activate venv
   - `python -m venv venv`
   - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install packages:
   - `pip install -r requirements.txt`
4. Build FAISS index:
   - `python build_index.py`
5. Start server:
   - `uvicorn app:app --reload --port 8000`
6. Test:
   - `curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"What are the symptoms of diabetes?"}'`

## Docker
- Build: `docker build -t healthcare-chatbot:latest .`
- Run: `docker run --rm -p 10000:10000 healthcare-chatbot:latest`

## Deploy to Render
1. Push repo to GitHub.
2. Sign up / sign in to Render.com.
3. Create a new **Web Service** and connect your GitHub repo.
4. Render will build from the Dockerfile automatically and deploy.
5. (No external API keys are required.)

## Project screenshot (for report)
The folder screenshot file is at:
`/mnt/data/a6cc4d0f-dd73-4cbe-b08e-b547e8ddd492.png`

Include that image path in your report or convert it to a URL if needed.
