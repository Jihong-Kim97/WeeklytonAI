import bs4
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

'''-------------RAG---------------'''
def rag():
    url = "https://namu.wiki/w/%EB%8B%B9%ED%99%A9(%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%20%EC%8B%9C%EB%A6%AC%EC%A6%88)"
    url2 ="https://namu.wiki/w/%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%202/%EC%A4%84%EA%B1%B0%EB%A6%AC"
    webloader = WebBaseLoader(web_path = [url, url2],
                                bs_kwargs=dict(
                                    parse_only = bs4.SoupStrainer(
                                        class_ = ("wiki-heading-content", "wiki-paragraph")
                                    )
                                ))

    data = webloader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 0
    )

    documents = text_splitter.split_documents(data)

    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True},
    )


    vectorstore = FAISS.from_documents(documents,
                                    embedding = embeddings_model,
                                    distance_strategy = DistanceStrategy.COSINE  
                                    )

    retriever = vectorstore.as_retriever()
    return retriever

'''--------------------------------'''