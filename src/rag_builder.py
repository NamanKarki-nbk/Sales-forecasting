import os 
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGBuilder:
    
    def __init__(self,csv_path:str, 
                vectorstore_path: str = "vector_store",
                embedding_model:str = "sentence-transformers/all-MiniLM-L6-v2",
                chunk_size:int=500,
                chunk_overlap:int=100
                ):
        
        self.csv_path = csv_path
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name= embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        #validating if the csv path exists or not
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found on path {csv_path}")
        
        
        #initialize the embeddings
        print("Initializing the embeddings: {embedding_model}\n")
        self.embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model_name)
        
        #initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap= self.chunk_overlap,
            separators=['\n\n', '\n', '.', " ", ""],
            length_function = len #ahile len use garya chu but for token based tokizer use gareko ramro
    
        )
        
        print('Rag builder is initialized')
        
    def load_documents(self)-> List[Document]:
        
        print(f"Loading Documents from {self.csv_path}")
        
        try:
            loader = CSVLoader(
                file_path=self.csv_path,
                encoding='utf-8'
            )
            
            documents = loader.load()
            print(f"Documents are loaded {len(documents)}")
            return documents
        
        except Exception as e:
            print(f"Error loading the document {str(e)}")
            raise

    def chunk_documents(self, documents : List[Document]):
        print(f"Splitting the text in chunk size of {self.chunk_size} with the text overlap of {self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created chunks of length {len(chunks)} from documents of size {len(documents)}")
        return chunks
    
    def create_vector_store(self, chunks: List[Document]):
        print("Creating vector embeddings and building FAISS index")
        
        try:
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding= self.embeddings 
            )
            print("Vector stores created")
            return vector_store
        
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            
    def save_vector_store(self,vector_store):
        os.makedirs(self.vectorstore_path, exist_ok=True)
        print(f"Saving vectors to: {self.vectorstore_path}")
        
        try:
            vector_store.save_local(self.vectorstore_path)
            
            #verify if the files were created or not
            index_file = os.path.join(self.vectorstore_path,"index.faiss")
            pkl_file = os.path.join(self.vectorstore_path, "index.pkl")
            
            if os.path.exists(index_file) & os.path.exists(pkl_file):
                print("Vector store saved successfully")
                
            else:
                print("Error in saving the file")
            
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            raise
    
    def build_vector_store(self):
        #load document
        document = self.load_documents()
        
        #chunking
        chunks = self.chunk_documents(document)
        
        #create the vector store
        vector_store = self.create_vector_store(chunks)
        
        #save the vectostore
        
        self.save_vector_store(vector_store)
        
        return vector_store
    
    
    
#testing purpose lagi matra ho:
    
if __name__ == "__main__":
        
        builder = RAGBuilder(
            csv_path="data/knowledge_base/downloadable_kb_csv.csv"
        )
        

        builder.build_vectorstore()