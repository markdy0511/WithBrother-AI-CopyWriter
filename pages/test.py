from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain_community.document_transformers import BeautifulSoupTransformer
import streamlit as st
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain  
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

llm = ChatOpenAI(temperature=0)

schema = {  
    "properties": { 
        "뉴스 제목": {"type": "string"},  
        "뉴스 요약": {"type": "string"},  
        "뉴스 전문": {"type": "string"}, 
        "작성일": {"type": "string"},  
        "작성자": {"type": "string"},   
        "언론사": {"type": "string"}, 
    },  
    "required": ["뉴스 제목", "뉴스 요약", "뉴스 전문", "작성일", "작성자", "언론사"],  
}  

def extract(content: str, schema: dict):  
    return create_extraction_chain(schema=schema, llm=llm).run(content)

bs_transformer = BeautifulSoupTransformer()
html2text_transformer = Html2TextTransformer()

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(  
chunk_size=1000, chunk_overlap=0  
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()

    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["article"])
    splits = splitter.split_documents(docs_transformed)

    extracted_contents = []
    for split in tqdm(splits):
        extracted_content = extract(content=split.page_content, schema=schema)
        extracted_contents.extend(extracted_content)
    print(extracted_contents)

    #transformed = html2text_transformer.transform_documents(docs)
    #st.write(docs)
    st.write(docs_transformed)