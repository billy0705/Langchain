from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA


class RAG():

    def __init__(self, db, pipeline):
        self.db = db
        self.pipeline = pipeline
        self.llm = None
        self.RQA = None

        self._create_llm()
        self._create_RQA()

    def _create_llm(self):
        self.llm = HuggingFacePipeline(pipeline=self.pipeline.pipe)

    def _create_RQA(self):
        self.RQA = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.db.as_retriever(),
            verbose=True
        )

    def qa(self, query):
        prompt = f"You are a chatbot who always responds in english! {query}"
        result = self.RQA.invoke(prompt)
        return result['result'].split("Helpful Answer: ")[1]
