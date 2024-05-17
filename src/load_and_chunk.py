import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from embedding_handler import embedding
from zenml import step

@step
class LoadAndChunk:


    def load_data(self):
        try:
            loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader) 
            data = loader.load()
            texts = [doc.page_content for doc in data]
            return texts
        except Exception as e:
            logging.error(f"An error occurred while load data: {e}")
            raise e

 
    def create_chunk(self):
        try:
            texts = self.load_data()
            if texts is None:
                raise ValueError("No texts to process.")

            embeddings = embedding()

            text_splitter = SemanticChunker( embeddings, breakpoint_threshold_type="percentile")
            docs = text_splitter.create_documents(texts)
            return docs
        except Exception as e:
            logging.error(f"An error occurred while creating chunks: {e}")
            raise e


# parent = LoadAndChunk()
# result = parent.create_chunk()
# print(result)
