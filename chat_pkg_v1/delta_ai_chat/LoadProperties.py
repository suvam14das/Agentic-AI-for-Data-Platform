class LoadProperties:

    def __init__(self):
        from delta_ai_chat.runtime_config import required_env

        self.model_name = required_env("DELTA_AI_LLM_MODEL_ID")
        self.endpoint = required_env("DELTA_AI_EMBEDDING_ENDPOINT")
        self.compartment_ocid = required_env("DELTA_AI_EMBEDDING_COMPARTMENT_ID")
        self.embedding_model_name = required_env("DELTA_AI_EMBEDDING_MODEL_ID")
        self.langchain_key = ""
        self.langchain_endpoint = ""

    def getModelName(self):
            return self.model_name

    def getEndpoint(self):
            return self.endpoint

    def getCompartment(self):
            return self.compartment_ocid

    def getEmbeddingModelName(self):
            return self.embedding_model_name

    def getLangChainKey(self):
            return self.langchain_key

    def getlangChainEndpoint(self):
            return self.langchain_endpoint











