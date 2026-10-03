import os
from delta_ai_chat.runtime_config import authenticate_session, required_env

from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_oci.embeddings import OCIGenAIEmbeddings

def generate_vector_store():
    profile_name = os.environ.get("DELTA_AI_PROFILE", "DEFAULT")

    try:
        # Initialize OCI Embeddings
        oci_embeddings = OCIGenAIEmbeddings(
            model_id=required_env("DELTA_AI_EMBEDDING_MODEL_ID"),
            service_endpoint=required_env("DELTA_AI_EMBEDDING_ENDPOINT"),
            compartment_id=required_env("DELTA_AI_EMBEDDING_COMPARTMENT_ID"),
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name
        )

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        general_txt_loader = DirectoryLoader(
            path= os.path.join(os.path.dirname(os.path.abspath(__file__)), "general_docs/"), 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            show_progress=True
        )
        general_txt_documents = general_txt_loader.load()
        general_chunks = text_splitter.split_documents(general_txt_documents)

        schema_csv_loader = DirectoryLoader(
            path= os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema_docs/"), 
            glob="**/*.xlsx", 
            loader_cls=UnstructuredExcelLoader,
            loader_kwargs={"mode": "elements"},
            show_progress=True
        )

        schema_documents = schema_csv_loader.load()
        # schema_chunks = text_splitter.split_documents(schema_documents)

        chunks = general_chunks + schema_documents

        vectorstore = FAISS.from_documents(chunks, oci_embeddings)
        vectorstore.save_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"))
        
    except Exception as e:
        error_str = str(e)
        if '401' in error_str:
            print(f"401 error detected during embedding initialization. Re-authenticating...")
            authenticate_session(os.environ.get("DELTA_AI_PROFILE", "DEFAULT"), required=True)
            oci_embeddings = OCIGenAIEmbeddings(
            model_id=required_env("DELTA_AI_EMBEDDING_MODEL_ID"),
            service_endpoint=required_env("DELTA_AI_EMBEDDING_ENDPOINT"),
            compartment_id=required_env("DELTA_AI_EMBEDDING_COMPARTMENT_ID"),
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile=profile_name
            )
            generate_vector_store()
        else:
            print(f"Error during embedding initialization: {error_str}")
            return

if __name__ == "__main__":
    generate_vector_store()
