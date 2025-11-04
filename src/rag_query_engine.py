import os 
from typing import Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGQueryEngine:
    
    def __init__(
        self,
        vectorstore_path:str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str="mistral",
        top_k = 3
    ):
        
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.top_k = top_k
        
        
        #initializing components
        
        self.initialize_embeddings()
        self.load_vector_store()
        self.initialize_llm()
        self.build_chain()

        print(f"Rag Engine is initialized with {self.llm_model_name}")
        
        
    def initialize_embeddings(self):
        print(f"Loading Embediings: {self.embedding_model_name}")
        self.embedding = HuggingFaceEmbeddings(
            model_name = self.embedding_model_name
        )
        
    
    def load_vector_store(self):
        vectorized_dir = self.vectorstore_path
        
        #yedi full path ayeni root path nai line
        if vectorized_dir.endswith('.faiss') or vectorized_dir.endswith('.pkl'):
            vectorized_dir = os.path.dirname(vectorized_dir)
        
        
        #check if the path exists or not
        if not os.path.exists(vectorized_dir):
            raise FileNotFoundError(
                f"Vector Store directory not found at {vectorized_dir} please create it"
            )
        
        #check if the files exists inside the dir or not
        index_file = os.path.join(vectorized_dir, 'index.faiss')
        pkl_file = os.path.join(vectorized_dir, 'index.pkl')
        
        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(
                f"Vector Store files not found in {vectorized_dir}, please create it"
            )
        
        print(f"Loading Vector Store from {vectorized_dir}")
        
        self.vector_store= FAISS.load_local(
            vectorized_dir,
            self.embedding,
            allow_dangerous_deserialization=True
        )
        
        #retriever for configurable paremters
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": self.top_k}
        )
        print(f"Vector store loaded (retrieving top {self.top_k} documents)") 
    
    def initialize_llm(self):
        
        print(f"Loading LLM model : {self.llm_model_name}")
        
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(model=self.llm_model_name)
        except ImportError:
            self.llm = Ollama(
                model=self.llm_model_name,
                temperature=0.1)
            
    def build_chain(self):
        
        #prompt template for the chain
        template = """You are a knowledgeable assistant with access to a holiday and sales knowledge base.
                    Your task is to answer questions accurately using the provided context.

                    Guidelines:
                    - Use ONLY the information from the context below
                    - If the context doesn't contain enough information, say 'I don't have enough information to answer that'
                    - Be concise and clear in your responses
                    - Cite specific details from the context when relevant

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
            
        #creating the prompt template
        self.prompt = PromptTemplate.from_template(template)
        
        #building rag chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG Chain is built successfully")
        
    def ask(self, query, return_context = False) :
        if not query.strip():
            return "Please provide a valid question."
        
        try:
            # get retrived documents
            retrieved_docs = self.retriever.invoke(query)
            
            # get the answer from the chain
            answer = self.chain.invoke(query)
            if return_context:
                    return {
                    "answer": answer,
                    "context_docs": retrieved_docs,
                    "query": query,
                    "num_docs": len(retrieved_docs)
                }
            
            return answer
    
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg
        
    
    
    def get_relevant_documents(self, query: str, k: Optional[int] = None):
        
        if k is not None:
            # Temporarily override retriever settings
            temp_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            return temp_retriever.invoke(query)
        
        return self.retriever.invoke(query)
    
    def similarity_search(self, query: str, k: int = 3):
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    
    
    
    
if __name__ == "__main__":
    rag_engine = RAGQueryEngine(
        vectorstore_path="vector_store",  
        llm_model="mistral",
        top_k=3
    )
    
    
    queries = [
        "How are sales affected in october and why",
    
    ]
    
    print("\n" + "="*60)
    print("RAG QUERY EXAMPLES")
    print("="*60)
    
    for query in queries:
        print(f"\nQ: {query}")
        answer = rag_engine.ask(query)
        print(f"A: {answer}")
        print("-" * 60)