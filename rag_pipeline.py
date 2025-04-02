from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
def chunk_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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

def query_rag(vector_stores, query):
    # Combine relevant chunks from all vector stores
    all_relevant_chunks = []
    for pdf_name, vector_store in vector_stores.items():
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        # Use invoke instead of get_relevant_documents
        relevant_chunks = retriever.invoke(query)
        for chunk in relevant_chunks:
            chunk.page_content = f"[{pdf_name}] {chunk.page_content}"
        all_relevant_chunks.extend(relevant_chunks)
    
    # Clean and limit chunks
    cleaned_chunks = clean_context(all_relevant_chunks[:5])
    context = " ".join(cleaned_chunks)
    
    # Truncate context to fit model’s max length (1024 tokens)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokens = tokenizer.encode(context, truncation=True, max_length=900)  # Leave room for prompt
    truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Initialize the model
    llm = HuggingFacePipeline.from_model_id(
        model_id="facebook/bart-large",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 50, "do_sample": False}  # No temperature needed
    )
    # Structured prompt
    prompt = (
        f"Question: {query}\n"
        f"Context: {truncated_context}\n"
        "Answer the question concisely and accurately based only on the context. "
        "Do not include random or irrelevant text:"
    )
    
    # Use invoke instead of __call__
    try:
        response = llm.invoke(prompt)
        if response.startswith("Answer:"):
            response = response[len("Answer:"):].strip()
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"
    
# Exception catched
if __name__ == "__main__":
    sample_text = "This is a test document about AI systems."
    chunks = chunk_text(sample_text)
    vector_store = create_vector_store(chunks)
    response = query_rag(vector_store, "What is this document about?")
    print(response)