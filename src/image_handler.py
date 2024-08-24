import torch
from PIL import Image
import faiss
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel

def retrieve_with_image(processor: CLIPProcessor, model: CLIPModel, faiss_path: str, query_image_path: str):
    """
    Retrieves the top similar images to the query image using a pre-built FAISS index.

    Parameters:
    processor (object): An instance of a pre-trained image processor.
    model (object): An instance of a pre-trained image model.
    faiss_path (str): The directory path where the FAISS index and image file paths are stored.
    query_image_path (str): The file path of the query image.

    Returns:
    tuple: A tuple containing two elements. The first element is a list of file paths of the top similar images,
           and the second element is None. In case of an exception, it prints the error message and returns None.
    """
    try:
        index, image_files = load_vision_index(faiss_path)
        similar_images, distances = retrieve_similar_images(processor, model, query_image_path, index, image_files)
        result = []
        for image, distance in zip(similar_images, distances):
            # print(f"Similar image: {image}, Distance: {distance}")
            result.append(image)
        return result, None
    except Exception as e:
        print(f"Error while retrieving similar images: {e}")

def create_index_for_image(faiss_path: str, vision_processor: CLIPProcessor, vision_model: CLIPModel, image_paths: str):
    """
    Creates a FAISS index for the given image paths using a pre-trained vision processor and model.
    The index is then saved to disk for future use.

    Parameters:
    faiss_path (str): The directory path where the FAISS index and image file paths will be stored.
    vision_processor (object): An instance of a pre-trained image processor.
    vision_model (object): An instance of a pre-trained image model.
    image_paths (list): A list of file paths for the images to be indexed.

    Returns:
    None
    """
    index, image_paths = build_index(vision_processor, vision_model, image_paths)
    save_index(faiss_path, index, image_paths)

def get_image_paths(data_source: str):
    """
    Retrieves a list of image file paths from a given directory.

    Parameters:
    data_source (str): The directory path where the images are located.

    Returns:
    list: A list of file paths for all image files (PNG, JPG, JPEG) found in the specified directory.
    """
    image_files = [os.path.join(data_source, f) for f in os.listdir(data_source) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

def image_to_vector(processor: CLIPProcessor, model: CLIPModel, image_path: str):
    """
    Converts an image at the specified file path into a vector representation using a pre-trained image processor and model.

    Parameters:
    processor (object): An instance of a pre-trained image processor. This processor should be compatible with the model.
    model (object): An instance of a pre-trained image model. This model should be compatible with the processor.
    image_path (str): The file path of the image to be converted into a vector.

    Returns:
    numpy.ndarray: A 1D numpy array representing the vectorized form of the input image.
    """
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    return image_features.squeeze().numpy()

def build_index(processor: CLIPProcessor, model: CLIPModel, image_paths: list):
    """
    Builds a FAISS index for the given image paths using a pre-trained vision processor and model.

    Parameters:
    processor (object): An instance of a pre-trained image processor. This processor should be compatible with the model.
    model (object): An instance of a pre-trained image model. This model should be compatible with the processor.
    image_paths (list): A list of file paths for the images to be indexed.

    Returns:
    tuple: A tuple containing two elements. The first element is the FAISS index built for the image vectors,
           and the second element is a list of the input image file paths.
    """
    image_vectors = []
    for image_path in image_paths:
        vector = image_to_vector(processor, model, image_path)
        image_vectors.append(vector)

    image_vectors = np.array(image_vectors).astype('float32')

    dimension = image_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(image_vectors)
    return index, image_paths

def save_index(faiss_path: str, index: faiss.IndexFlatL2, image_files: list, index_name="image_vectors.index", image_files_name="image_files.txt"):
    """
    Saves a FAISS index and corresponding image file paths to disk.

    Parameters:
    faiss_path (str): The directory path where the FAISS index and image file paths will be stored.
    index (faiss.Index): The FAISS index to be saved.
    image_files (list): A list of file paths for all images in the dataset.
    index_name (str, optional): The name of the FAISS index file. Default is "image_vectors.index".
    image_files_name (str, optional): The name of the text file containing image file paths. Default is "image_files.txt".

    Returns:
    None
    """
    index_path = os.path.join(faiss_path, index_name)
    image_files_path = os.path.join(faiss_path, image_files_name)
    faiss.write_index(index, index_path)
    with open(image_files_path, "w") as f:
        for image_file in image_files:
            f.write(image_file + "\n")

def load_vision_index(faiss_path: str, index_name="image_vectors.index", image_files_name="image_files.txt"):
    """
    Load a pre-built FAISS index and corresponding image file paths from disk.

    Parameters:
    faiss_path (str): The directory path where the FAISS index and image file paths are stored.
    index_name (str, optional): The name of the FAISS index file. Default is "image_vectors.index".
    image_files_name (str, optional): The name of the text file containing image file paths. Default is "image_files.txt".

    Returns:
    tuple: A tuple containing two elements. The first element is the loaded FAISS index,
           and the second element is a list of image file paths corresponding to the index.
    """
    index_path = os.path.join(faiss_path, index_name)
    image_files_path = os.path.join(faiss_path, image_files_name)
    index = faiss.read_index(index_path)
    with open(image_files_path, "r") as f:
        image_files = f.read().splitlines()
    return index, image_files

def retrieve_similar_images(processor: CLIPProcessor, model: CLIPModel, query_image_path: str, index: faiss.IndexFlatL2, image_files: list, top_k=3):
    """
    Retrieve the top k most similar images to the query image using a pre-built FAISS index.

    Parameters:
    processor (object): An instance of a pre-trained image processor.
    model (object): An instance of a pre-trained image model.
    query_image_path (str): The file path of the query image.
    index (faiss.Index): The pre-built FAISS index for image vectors.
    image_files (list): A list of file paths for all images in the dataset.
    top_k (int, optional): The number of top similar images to retrieve. Default is 3.

    Returns:
    tuple: A tuple containing two lists. The first list contains the file paths of the top k similar images,
           and the second list contains the corresponding distances between the query image and the similar images.
    """
    query_vector = image_to_vector(processor, model, query_image_path)
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    similar_images = [image_files[i] for i in indices[0]]
    return similar_images, distances[0]