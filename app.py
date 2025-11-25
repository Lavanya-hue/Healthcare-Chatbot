import json
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Healthcare Chatbot with NER & Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
INDEX_PATH = "index.faiss"
META_PATH = "index_meta.json"
TOP_K = 5

print("Loading embedding model...")
embed = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
meta = json.load(open(META_PATH, "r", encoding="utf-8"))
docs = meta["docs"]

print("Loading NER pipeline...")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

print("Loading zero-shot classification pipeline...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["symptoms", "disease definition", "treatment", "prevention", "general information"]

print("Loading generator model...")
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)


class Query(BaseModel):
    question: str
    max_new_tokens: int = 150


def expand_query(query):
    """Expand query with synonyms for better retrieval."""
    synonyms = {
        "dengue": "dengue fever viral infection mosquito symptoms treatment",
        "fever": "fever temperature high body heat paracetamol medicine",
        "headache": "headache head pain migraine tension",
        "paracetamol": "paracetamol acetaminophen fever pain medication safe",
        "diabetes": "diabetes blood sugar glucose symptoms complications chest pain",
        "chest pain": "chest pain heart cardiac diabetes emergency doctor",
        "symptom": "symptom sign manifestation disease",
        "mother": "family member relative",
        "migraine": "migraine headache neurological",
    }
    expanded = query.lower()
    for key, value in synonyms.items():
        if key in expanded:
            expanded += " " + value
            break  # Only expand once
    return expanded


def retrieve_context(query, top_k=5):
    """Retrieve relevant docs with better filtering."""
    expanded = expand_query(query)
    q_emb = embed.encode([expanded], convert_to_numpy=True)
    q_emb = q_emb.astype(np.float32)
    
    # Search with increased k to get more candidates
    D, I = index.search(q_emb, min(top_k * 2, len(docs)))
    
    results = []
    scores = []
    
    for idx, distance in zip(I[0], D[0]):
        if 0 <= idx < len(docs):
            results.append(docs[idx])
            scores.append(float(distance))
    
    # Return top results (don't filter out by score threshold)
    return results[:top_k], scores[:top_k]


@app.post("/ask")
def ask(payload: Query):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        entities = ner(question)
    except:
        entities = []
    
    try:
        cls = classifier(question, candidate_labels=candidate_labels)
        query_type = cls["labels"][0]
    except:
        query_type = "general information"

    retrieved, scores = retrieve_context(question, top_k=5)
    context_text = "\n\n".join(retrieved)

    # Remove strict filtering - use retrieval results as-is
    if not context_text.strip():
        safe_answer = (
            "I don't have specific information about that in my knowledge base. "
            "Please consult a qualified healthcare professional."
        )
        return {
            "answer": safe_answer,
            "query_type": query_type,
            "entities": entities,
            "retrieved_context": []
        }

    prompt = f"""You are a helpful healthcare assistant. Answer based on the provided medical information.

MEDICAL CONTEXT:
{context_text}

USER QUESTION:
{question}

INSTRUCTIONS:
- Provide a clear, helpful answer based on the context.
- If the context answers the question, provide that answer directly.
- Always recommend consulting a healthcare professional for serious symptoms.
- Do not say "I don't know" if context is provided.

ANSWER:"""

    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = gen_model.generate(
        **inputs,
        max_new_tokens=payload.max_new_tokens,
        temperature=0.6,
        do_sample=False
    )
    answer = gen_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    if not answer:
        answer = "Please consult a healthcare professional for personalized medical advice."

    return {
        "answer": answer,
        "query_type": query_type,
        "entities": entities,
        "retrieved_context": retrieved
    }