from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import re

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        _model = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 100, "do_sample": False}
        )
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return _model, _tokenizer

def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Larger chunks, less overlap
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def clean_context(chunks):
    cleaned = []
    for chunk in chunks:
        lines = chunk.page_content.split("\n")
        cleaned_lines = [line for line in lines if line.strip() and not line.strip().isdigit()]
        cleaned.append(" ".join(cleaned_lines))
    return cleaned

def clean_response(response, query):
    query_lower = query.lower()
    if "how much" in query_lower or "what is" in query_lower or "cash paid" in query_lower and "explain" not in query_lower:
        match = re.search(r'\$?\s*[\d,]+(?:\.\d+)?', response)
        if match:
            value = match.group(0).replace(" ", "")
            if not value.startswith("$"):
                value = f"${value}"
            return value
        return response.strip()
    else:
        response = " ".join(response.split())
        if not response.endswith("."):
            response += "."
        if len(response) > 100:
            response = response[:97] + "..."
        return response

def query_rag(vector_stores, query):
    all_relevant_chunks = []
    for pdf_name, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Fewer chunks for speed
        relevant_chunks = retriever.invoke(query)
        for chunk in relevant_chunks:
            chunk.page_content = f"[{pdf_name}] {chunk.page_content}"
        all_relevant_chunks.extend(relevant_chunks)
    
    cleaned_chunks = clean_context(all_relevant_chunks[:3])  # Limit to 3 chunks
    context = " ".join(cleaned_chunks)
    
    llm, tokenizer = get_model()
    tokens = tokenizer.encode(context, truncation=True, max_length=400)  # Lower limit for speed
    truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    if "explain" in query.lower():
        prompt = f"Q: {query}\nC: {truncated_context}\nExplain concisely (<100 chars):"
    else:
        prompt = f"Q: {query}\nC: {truncated_context}\nExtract value (e.g., $18,651):"
    
    try:
        response = llm.invoke(prompt)
        return clean_response(response, query)
    except Exception as e:
        return f"Error generating answer: {str(e)}"