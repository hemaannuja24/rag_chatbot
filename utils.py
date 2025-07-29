import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import tempfile

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_vectorstore_from_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)
    return vectorstore
