import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_vectorstore_from_text(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)
    return vectorstore
