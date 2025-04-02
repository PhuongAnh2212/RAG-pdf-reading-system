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

def clean_response(response, query):
    # Detect if the query asks for a numeric value or an explanation
    query_lower = query.lower()
    if "how much" in query_lower or "what is" in query_lower or "cash paid" in query_lower and "explain" not in query_lower:
        # Extract numeric value with currency formatting
        match = re.search(r'\$?\s*[\d,]+(?:\.\d+)?', response)
        if match:
            value = match.group(0).replace(" ", "")
            if not value.startswith("$"):
                value = f"${value}"
            return value
        return response.strip()
    else:
        # Return explanatory text, cleaned up
        response = response.strip()
        # Remove redundant whitespace and ensure it ends with a period
        response = " ".join(response.split())
        if not response.endswith("."):
            response += "."
        # Limit to a reasonable length (e.g., 100 characters) for brevity
        if len(response) > 100:
            response = response[:97] + "..."
        return response

def query_rag(vector_stores, query):
    all_relevant_chunks = []
    for pdf_name, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_chunks = retriever.invoke(query)
        for chunk in relevant_chunks:
            chunk.page_content = f"[{pdf_name}] {chunk.page_content}"
        all_relevant_chunks.extend(relevant_chunks)
    
    cleaned_chunks = clean_context(all_relevant_chunks[:5])
    context = " ".join(cleaned_chunks)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokens = tokenizer.encode(context, truncation=True, max_length=900)
    truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100, "do_sample": False}  # Increased tokens for explanations
    )
    
    # Flexible prompt based on query type
    if "explain" in query.lower():
        prompt = (
            f"Question: {query}\n"
            f"Context: {truncated_context}\n"
            "Provide a concise explanation based on the context. Include relevant data points and keep it under 100 characters if possible:"
        )
    else:
        prompt = (
            f"Question: {query}\n"
            f"Context: {truncated_context}\n"
            "Extract the exact value or data point requested from the context. Return only the relevant number or text, formatted clearly (e.g., $18,651):"
        )
    
    try:
        response = llm.invoke(prompt)
        if response.startswith("Answer:"):
            response = response[len("Answer:"):].strip()
        return clean_response(response, query)
    except Exception as e:
        return f"Error generating answer: {str(e)}"