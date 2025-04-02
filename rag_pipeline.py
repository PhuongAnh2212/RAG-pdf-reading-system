from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import re

def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

def clean_context(chunks):
    cleaned = []
    for chunk in chunks:
        lines = chunk.page_content.split("\n")
        cleaned_lines = [
            line for line in lines 
            if line.strip() 
            and not line.strip().isdigit() 
            and "accompanying notes" not in line.lower()
        ]
        cleaned.append(" ".join(cleaned_lines))
    return cleaned

def clean_response(response, query, source_chunk):
    query_lower = query.lower()
    match = re.search(r'\$?\s*[\d,]+(?:\.\d+)?', response)
    value = match.group(0).replace(" ", "") if match else response.strip()
    if value and not value.startswith("$"):
        value = f"${value}"
    
    source_match = re.search(r'\[(.*?)\]', source_chunk)
    source = source_match.group(1) if source_match else "unknown source"
    
    purpose = query_lower.replace("what is", "").replace("how much", "").strip()
    if "explain" in query_lower:
        response = " ".join(response.split())
        if not response.endswith("."):
            response += "."
        if len(response) > 100:
            response = response[:97] + "..."
        return f"{response} (from {source})"
    else:
        return f"{value} for {purpose} from {source}"

def query_rag(vector_stores, query):
    all_relevant_chunks = []
    for pdf_name, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_chunks = retriever.invoke(query)
        for chunk in relevant_chunks:
            chunk.page_content = f"[{pdf_name}] {chunk.page_content}"
        all_relevant_chunks.extend(relevant_chunks)
    
    if not all_relevant_chunks:
        return "No relevant data found."
    
    cleaned_chunks = clean_context(all_relevant_chunks[:5])
    context = " ".join(cleaned_chunks)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokens = tokenizer.encode(context, truncation=True, max_length=800)
    truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100, "do_sample": False}
    )
    
    # Prompt with instruction to include relevant details
    if "explain" in query.lower():
        prompt = (
            f"Q: {query}\n"
            f"C: {truncated_context}\n"
            "Explain concisely (<100 chars) with key data:"
        )
    else:
        prompt = (
            f"Q: {query}\n"
            f"C: {truncated_context}\n"
            "Extract value (e.g., $18,651) with context:"
        )
    
    prompt_tokens = tokenizer.encode(prompt, truncation=False)
    if len(prompt_tokens) > 1024:
        excess = len(prompt_tokens) - 1024 + 100
        tokens = tokenizer.encode(truncated_context, truncation=True, max_length=800-excess)
        truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
        prompt = (
            f"Q: {query}\n"
            f"C: {truncated_context}\n"
            "Extract value (e.g., $18,651) with context:" if "explain" not in query.lower() else
            f"Q: {query}\n"
            f"C: {truncated_context}\n"
            "Explain concisely (<100 chars) with key data:"
        )
    
    try:
        response = llm.invoke(prompt)
        if response.startswith("Answer:"):
            response = response[len("Answer:"):].strip()
        source_chunk = all_relevant_chunks[0].page_content if all_relevant_chunks else ""
        return clean_response(response, query, source_chunk)
    except Exception as e:
        return f"Error generating answer: {str(e)}"