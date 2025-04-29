from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os


# LLM model definition
llm_model = "llama3"


# Chromadb configuration
# ChromaDB client initialization with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))


# Custom embedding function definition for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # The input must be in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)


# Embedding function initialization with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)


# Collection definition for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)


# Documents addition to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )


# The external data to be used in our case is a collection of books with non european origins

raw_text = """
    
    - Premier œuvre: Épopée de Gilgamesh (Mésopotamie, env. 2100 av. J.-C.)
    Considérée comme la plus ancienne œuvre littéraire connue, cette épopée sumérienne explore 
    des thèmes universels tels que l'amitié, la quête d'immortalité et la condition humaine.

    - Deuxième oeuvre: Shijing (Classique des poèmes) (Chine, XIe–VIIe siècle av. J.-C.)
    Ce recueil de 305 poèmes est l'une des œuvres fondatrices de la littérature chinoise, 
    influençant la poésie et la pensée confucéenne pendant des siècles.
 
    - Troisième œuvre: Le Livre des ruses (Kitāb al-ḥiyal) 
    Rédigé au IXe siècle par les Banū Mūsā, 
    ce recueil présente des dispositifs mécaniques ingénieux, illustrant l'ingéniosité scientifique 
    du monde islamique médiéval.

    - Quatrième œuvre: Le Muqaddimah d’Ibn Khaldun - écrit au XIVe siècle, est un texte fondamental de l’historiographie islamique. 
    Ibn Khaldun, souvent considéré comme le père de la sociologie, y développe des concepts 
    révolutionnaires sur la civilisation, la montée et la chute des empires, et l’importance de la 
    religion dans l’organisation sociale. Ce livre reste un pilier pour comprendre l’histoire islamique 
    et mondiale​.

    - Cinquième œuvre: Recueil de hadiths, ce texte constitue l’une des plus grandes œuvres de spiritualité islamique ans l'histoire. 
    Imam Nawawi y compile des paroles prophétiques organisées par thèmes comme la patience, la piété, 
    et la générosité. Ce livre est incontournable pour ceux qui cherchent à approfondir leur 
    connaissance des enseignements du Prophète​.

    - Sixième œuvre: Les Upanishads (Inde, env. 800–200 av. J.-C.)
    Textes philosophiques majeurs de l'hindouisme, les Upanishads abordent des concepts 
    métaphysiques profonds tels que le Brahman (l'absolu) et l'Atman (le soi).

    - Semptième œuvre: Things Fall Apart de Chinua Achebe (Nigeria, 1958)
    Ce roman est considéré comme le livre le plus important de la littérature africaine moderne. 
    Il a été traduit en 57 langues et s'est vendu à plus de 20 millions d'exemplaires.

    - Huitième œuvre: Le Prophète de Khalil Gibran (Liban/États-Unis, 1923)
    Ce recueil de textes poétiques est devenu l'un des livres les plus vendus de tous les temps, 
    traduit en plus de 100 langues.

    - Neufième œuvre: Cent ans de solitude de Gabriel García Márquez (Colombie, 1967)
    Ce roman emblématique du réalisme magique a valu à son auteur le prix Nobel de littérature en 1982.

    - Dixième œuvre: Mon nom est Rouge d'Orhan Pamuk (Turquie, 1998)
    Ce roman, qui mêle traditions narratives orientales et techniques postmodernes, a contribué à faire de 
    Pamuk le premier Turc à recevoir le prix Nobel de littérature en 2006. 

    - Onzièe œuvre: La Végétarienne de Han Kang (Corée du Sud, 2007)
    Ce roman, couronné par l'International Booker Prize, explore les traumatismes historiques et 
    la fragilité de la vie humaine. Han Kang a reçu le prix Nobel de littérature en 2024. 
    """


# Step1: Text splitting in fragments
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Each chunk will have ~1000 characters
    chunk_overlap=100      # To maintain continuity between fragments
)

chunks = text_splitter.split_text(raw_text)
chunk_ids = [f"doc1_chunk_{i}" for i in range(len(chunks))]

# Step 2: Add chunks to chromadb
add_documents_to_collection(chunks, chunk_ids)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combinning ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Relevant documents retreival from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(doc[0] for doc in retrieved_docs) if retrieved_docs else "Aucun document pertinent trouvé."

    # Step 2: Query sending along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuéstion: {query_text}\nRéponse:"
    print("---------- Prompt Augmenté ----------")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response


# Example usage
# Query definition
query = "Quelles sont les œuvres littéraires les plus influentes de l'histoire du monde? \
Réponds à la question suivante en t'appuyant sur ce contexte ainsi que sur tes connaissances internes. \
Si tu peux compléter ou enrichir les informations du contexte, fais-le de manière pertinente et en français."


response = rag_pipeline(query)
print("---------- Réponse du LLM ----------n", response)