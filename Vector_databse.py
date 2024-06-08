from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


class Vector_DB():
    '''
    A class creating a vector database (Chroma).
    '''

    def __init__(
            self,
            embedding_model_name,
            device_map={'device': 'cpu'},
            db_path=None
            ):
        '''
        initial vectordb class

        Args:
        embedding_model_name (str): embedding model from huggingface
        device_map (dict) defult={'device': 'cpu'}
            : device use for embedding calculation
        db_path (str) defult=None: persist database path
        '''
        self.embedding_model_name = embedding_model_name
        self.device_map = device_map
        self.db_path = db_path
        self.db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=100,
                                chunk_overlap=5
                            )
        # self.pdf_folder_path = pdf_folder_path

        self.embedding = self._load_embedding()

    def _load_embedding(self):
        '''
        get embedding model from hugging face
        '''
        print("Loading embedding model.")
        return HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs=self.device_map
                )

    def persist_db(self):
        '''
        load persist data base from the path
        '''
        print("Loading persist database.")
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding
        )

    def _get_documents_list(self, folder_path):
        '''
        get a list of documents store in the folder

        Args:
        folder_path (str): path of the document folder

        Return:
        docs_list (list[str]): A list to store all the documents' path
        '''
        docs_list = []
        files_list = os.listdir(folder_path)
        for file_name in files_list:
            file_path = os.path.join(folder_path, file_name)
            docs_list.append(file_path)

        return docs_list

    def _get_source_list(self):
        '''
        get a list of sources store in the db

        Return:
        sources_list (list[str]): A list to store all the sources
        '''
        db_collections = self.db.get()
        sources_list = []
        for x in range(len(db_collections["ids"])):
            doc = db_collections["metadatas"][x]
            source = doc["source"]
            if source not in sources_list:
                sources_list.append(source)

        print("Sources: ", sources_list)
        return sources_list

    def _get_new_docs(self, folder_path):
        '''
        get a list of new documents which store in the path

        Args:
        folder_path (str): path of the document folder

        Return:
        new_docs_list (list[str]): A list to store all the new documents' path
        '''
        docs_list = self._get_documents_list(folder_path)
        sources_list = self._get_source_list()
        new_docs_list = []

        for file_path in docs_list:
            if file_path not in sources_list:
                new_docs_list.append(file_path)

        return new_docs_list

    def add_docs_from_path(self, folder_path):
        '''
        Add the document into the database

        Args:
        folder_path (str): path of the document folder
        '''
        assert self.db is not None
        new_docs_list = self._get_new_docs(folder_path)

        for doc_path in new_docs_list:
            if doc_path.endswith('.pdf'):
                loader = PyMuPDFLoader(doc_path)
                docs = loader.load()
                splits_docs = self.text_splitter.split_documents(docs)
                self.db.add_documents(splits_docs)
                print("Add document: ", doc_path[2:])

        self.db.persist()
        print("Database saved.")
