from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


def gen_documents_list(pdf_folder_path):
    docs_list = []
    pdf_folders = os.listdir(pdf_folder_path)
    for module_name in pdf_folders:
        module_pdf_folder_path = os.path.join(pdf_folder_path, module_name)
        module_pdf_list = os.listdir(module_pdf_folder_path)
        module_pdf_list.sort()
        for pdf_name in module_pdf_list:
            pdf_path = os.path.join(module_pdf_folder_path, pdf_name)
            docs_list.append(pdf_path)

    return docs_list


if __name__ == "__main__":
    pdf_folder_path = "./pdf_folder/"
    db_path = "./db"
    # docs_list = gen_documents_list(pdf_folder_path)
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'}
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs)
    # print(docu_list)

    loader = DirectoryLoader(pdf_folder_path, glob='./*.pdf',
                             loader_cls=PyMuPDFLoader,
                             show_progress=True)
    documents = loader.load()
    print(f'documents:{len(documents)}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                   chunk_overlap=5)
    splits_documents = text_splitter.split_documents(documents)
    print(f'documents:{len(splits_documents)}')

    docsearch = Chroma.from_documents(splits_documents, embedding,
                                      persist_directory=db_path)
    docsearch.persist()

    # docsearch = Chroma(persist_directory=db_path,
    #                    embedding_function=embedding)
