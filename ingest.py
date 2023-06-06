from dotenv import load_dotenv
from typing import Tuple, List, Dict, Callable
import PyPDF2
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """

    :param file_path:
    :return: A tuple containing metadata and chunks of pdf
    """

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def extract_metadata_from_pdf(file_path: str) -> Dict[str, str]:
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        metadata = pdf.metadata
        return {
            "title": metadata.title.strip()
        }


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((i, page.extract_text()))
        return pages


def clean_text(raw_pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]) -> List[
    Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in raw_pages:
        cleaned_page = None
        for cleaning_function in cleaning_functions:
            cleaned_page = cleaning_function(text)
        if cleaned_page:
            cleaned_pages.append((page_num, cleaned_page))
    return cleaned_pages


def merge_hyphened_words(text: str) -> str:
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)


def fix_newlines(text: str) -> str:
    return re.sub(r'(?<!\n)\n(?!\n)', r' ', text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r'\n{2,}', r'\n', text)


def text_to_docs(text: List[Tuple[int,str]], metadata: Dict[str, str]) -> List[Document]:
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200,
                                                       separators=['\n', '\n\n', '.', ',', '!', '?', ' ', '']
                                                       )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'page_number': page_num,
                    'chunk': i,
                    'source': f"p{page_num}-{i}",
                    **metadata
                })
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == '__main__':
    load_dotenv()

    # step 1: Parse pdf
    file_path = './data/book1.pdf'
    raw_pages, metadata = parse_pdf(file_path)

    cleaning_functions = [
        merge_hyphened_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    try:
        print(f'Embedding model cuda: {embeddings.client.is_cuda()}')
    except Exception:
        pass

    vector_store = Chroma.from_documents(document_chunks,
                                         embeddings,persist_directory='./data/chroma',
                                         anonymous_telemetry=True,
                                         chroma_db_impl="duckdb+parquet"
                                         )

    vector_store.persist()

