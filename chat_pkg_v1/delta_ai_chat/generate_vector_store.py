import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OCIGenAIEmbeddings
try :
    from delta_ai_chat.LoadProperties import LoadProperties
except ImportError as e:
    from LoadProperties import LoadProperties
import subprocess

def generate_vector_store():
    properties = LoadProperties()

    try:
        # Initialize OCI Embeddings
        oci_embeddings = OCIGenAIEmbeddings(
            model_id=properties.getEmbeddingModelName(),
            service_endpoint=properties.getEndpoint(),
            compartment_id=properties.getCompartment(),
            model_kwargs={"truncate": True},
            auth_type="SECURITY_TOKEN",
            auth_profile="bmc-sie-prod"
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
            subprocess.run("oci session authenticate --profile-name bmc-sie-prod --region us-ashburn-1 --tenancy-name bmc_operator_access --auth security_token", shell=True, check=True)
            oci_embeddings = OCIGenAIEmbeddings(
                model_id=properties.getEmbeddingModelName(),
                service_endpoint=properties.getEndpoint(),
                compartment_id=properties.getCompartment(),
                model_kwargs={"truncate": True},
                auth_type="SECURITY_TOKEN",
                auth_profile="bmc-sie-prod"
            )
            generate_vector_store()
        else:
            print(f"Error during embedding initialization: {error_str}")
            return

if __name__ == "__main__":
    generate_vector_store()
