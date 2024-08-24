from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass, field
from typing import List
from src.document_handler import load_documents, split_into_chunks, create_index_for_documents, retrieve_with_llm
from src.image_handler import get_image_paths, create_index_for_image, retrieve_with_image


FAISS_PATH = "faiss"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "moondream"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
vision_model = CLIPModel.from_pretrained(VISION_MODEL_NAME)
vision_processor = CLIPProcessor.from_pretrained(VISION_MODEL_NAME)


@dataclass
class Data:
    chunks: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)

def preprocess_data(data_source):
    """
    Preprocess data from a given source to prepare for indexing.

    Parameters:
    data_source (str): The source of the data. It can be a directory path, a file path, or any other data source.

    Returns:
    Data: A Data object containing the preprocessed chunks of text and image paths.

    The function performs the following steps:
    1. Load documents from the data source using the load_documents function.
    2. Get image paths from the data source using the get_image_paths function.
    3. Split the loaded documents into chunks using the split_into_chunks function.
    4. Create a Data object with the chunks and image paths.
    5. Return the Data object.
    """
    documents = load_documents(data_source)
    image_paths = get_image_paths(data_source)
    chunks = split_into_chunks(documents)
    return Data(chunks, image_paths)

def create_index(data):
    """
    Creates an index for the given data, including both text and image data.

    Parameters:
    data (Data): A Data object containing preprocessed chunks of text and image paths.

    The function performs the following steps:
    1. If there are chunks of text in the data, it calls the create_index_for_documents function to create an index for the text data.
    2. If there are image paths in the data, it prints the image paths and calls the create_index_for_image function to create an index for the image data.
    """
    if len(data.chunks) > 0:
        create_index_for_documents(FAISS_PATH, EMBEDDING_MODEL, data.chunks)
    if len(data.image_paths) > 0:
        print(data.image_paths)
        create_index_for_image(FAISS_PATH, vision_processor, vision_model, data.image_paths)

def retrieve_documents(query):
    """
    Retrieves documents based on the given query.

    This function checks the file type of the query. If the query is an image (PNG, JPG, JPEG),
    it calls the `retrieve_with_image` function to retrieve documents using image-based search.
    Otherwise, it calls the `retrieve_with_llm` function to retrieve documents using LLM-based search.

    Parameters:
    query (str): The query string. It can be a text query or an image path.

    Returns:
    tuple: A tuple containing the retrieved documents and the context (if applicable).
           If the query is an image, the context is None.
           If the query is a text, the context is the original document that the LLM used for retrieval.
    """
    if query.endswith(('.png', '.jpg', '.jpeg')):
        response, context = retrieve_with_image(vision_processor, vision_model, FAISS_PATH, query)
    else:
        response, context = retrieve_with_llm(LLM_MODEL, EMBEDDING_MODEL, FAISS_PATH, query, is_retrieve_with_llm=True)
    return response, context

def detect_hallucinations(response, context):
    """
    Detects hallucinations in a response based on similarity with the context.

    This function uses OllamaEmbeddings to generate embeddings for the response and context,
    calculates the cosine similarity between them, and checks for common words between the response and context.
    It then determines if the response is a hallucination based on the similarity score and the presence of common words.

    Parameters:
    response (str): The response to be checked for hallucinations.
    context (str): The context or original document that the response is based on.

    Returns:
    dict: A dictionary containing the following keys:
          - "is_hallucination": A boolean indicating whether the response is a hallucination.
          - "similarity_score": The cosine similarity score between the response and context.
          - "common_words": A list of common words between the response and context (ignoring case).
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    response_vector = embeddings.embed_query(response)
    context_vector = embeddings.embed_query(context)
    similarity_score = cosine_similarity([response_vector], [context_vector])[0][0]
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    common_words = response_words.intersection(context_words)
    is_hallucination = similarity_score < 0.1 or len(common_words) == 0
    result = {
        "is_hallucination": is_hallucination,
        "similarity_score": similarity_score,
        "common_words": list(common_words)
    }

    return result

if __name__ == "__main__":
    data_source = "docs"
    # query = "Phi khoi tao he thong" # To check hallucination
    # query = "What is Amazon Bedrock?" # To documents
    query = "docs/cat_3.jpeg" # To retrieve image
    # data = preprocess_data(data_source)
    # create_index(data)
    response, context = retrieve_documents(query)
    if context is not None:
        print(f"{query}: {response['answer']}")
        hallucination_prediction_result = detect_hallucinations(response["answer"], context)
        print("Hallucination prediction result: ", hallucination_prediction_result)
    else:
        for i, doc in enumerate(response):
            print(f"Query result {i+1}:")
            if hasattr(doc, 'page_content'):
                print(doc.page_content)
            else:
                print(doc)
            print("---")
