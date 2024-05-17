from langchain_google_genai import GoogleGenerativeAIEmbeddings
from zenml import step
import os
from dotenv import load_dotenv
load_dotenv()


@step
def embedding(google_api_key):
    try:
        # Retrieve API key from environment variables
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        # Create Google Generative AI Embeddings instance
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
        return embeddings
    except Exception as e:
        logging.error(f"An error occurred while creating the embeddings: {e}")
        raise