import requests
from bs4 import BeautifulSoup
import asyncio
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser, output_parser
from langchain_community.document_loaders import YoutubeLoader
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader
from langchain.schema import StrOutputParser
from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer


strict_llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
) #요약 같은 정확한 업무를 해야 할 때

flexible_llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.2,
) #요약 같은 정확한 업무를 해야 할 때

mellting_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.5,
) #스크립트 작성 같은 창의력을 요하는 업무를 해야 할 때

bs_transformer = BeautifulSoupTransformer()

def load_naver_news(url): #네이버 뉴스에서 필요한 정보 추출
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser') #네이버 뉴스 html
    # 네이버 뉴스에서 제목, 작성 기관, 작성일, 작성자, 기사 내용을 추출    
    try:
        title = soup.select("#title_area > span")[0].get_text()
        ref = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_top._LAZY_LOADING_WRAP > a > img.media_end_head_top_logo_img.light_type._LAZY_LOADING._LAZY_LOADING_INIT_HIDE")[0].get("title")
        date = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")[0].get("data-date-time")
        maker = soup.select("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_journalist > a > em")[0].get_text()
        context = soup.select("#dic_area")[0].get_text()

        source = {
            "title" : title,
            "ref": ref,
            "date": date,
            "maker": maker,
            "context": context
        }
    except:
        loader = AsyncChromiumLoader([url])
        docs = loader.load()
        docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["article"])
        source = {
        "title" : None,
        "ref": None,
        "date": None,
        "maker": None,
        "context": docs_transformed[0].page_content
    }

    return source

def load_youtube_transcript(url): #유튜브 링크에서 필요한 정보 추출
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True,
        language=["en", "ko"],
        translation="ko")
    source = loader.load() #list 형태인 듯 [0:~~~, metadata={'source','title','description','view_count','thumbnail_url','publish_date','length','author'}]

    return source

def source_input_form():
    with st.form("Source_Info_form"):
        title = st.text_input("What title is it?")
        description = st.text_input("What do you want to tell to reader?")
        num_outline = st.text_input("How many outlines do you have?")
        submitted = st.form_submit_button("Set Writing Material")
    if submitted:
        return title, description, num_outline, submitted
    else:
        return None, None, None, None

class ListOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("list", "")
        return text.split("/")
    
list_output_parser = ListOutputParser()


st.set_page_config(
    page_title="Script Maker",
    page_icon="📜",
)


st.markdown(
    """
    # Script Maker
            
    Use Script Assistant for your magazine work.
            
    Start by writing the URL of the website or/with what you want to write on the sidebar.
"""
)


with st.sidebar: #원하는 소스를 만드는 곳
    submitted = None
    writing_material = {}
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "NAVER NEWS",
            "YOUTUBE VEDIO",
            "No Source"
        ),
    )

    if choice == "NAVER NEWS":
        url = st.text_input(
        "Write down a URL",
        placeholder="https://n.news.naver.com/article/~~~",
        )
        if url:
            with st.spinner('Loading Naver News Website...'):
                TITLE, DESCRIPTION, NUM_OUTLINE, submitted = source_input_form()
                if submitted:
                    source = load_naver_news(url) # dictionary
                    writing_material = source
                    writing_material["magazine_title"] = TITLE
                    writing_material["magazine_context"] = DESCRIPTION
                    writing_material["num_magazine_outline"] = NUM_OUTLINE


    elif choice == "YOUTUBE VEDIO":
        url = st.text_input(
        "Write down a URL",
        placeholder="https://www.youtube.com/~~~",
        )
        if url:
            with st.spinner('Loading Youtube Website...'):
                TITLE, DESCRIPTION, NUM_OUTLINE, submitted = source_input_form()
                if submitted:
                    source = load_youtube_transcript(url) # list [0:~~~,metadata={'source','title','description','view_count','thumbnail_url','publish_date','length','author'}]
                    writing_material = {
                    "title" : source[0].metadata['title'],
                    "ref": None,
                    "date": source[0].metadata['publish_date'],
                    "maker": source[0].metadata['author'],
                    "context": source[0].page_content
                    }
                    writing_material["magazine_title"] = TITLE
                    writing_material["magazine_context"] = DESCRIPTION
                    writing_material["num_magazine_outline"] = NUM_OUTLINE
    else:
        st.write("no source Done")
        source = "Nothing"
        TITLE, DESCRIPTION, NUM_OUTLINE, submitted = source_input_form()
        writing_material = {
                "title" : None,
                "ref": None,
                "date": None,
                "maker": None,
                "context": None
                }
        writing_material["magazine_title"] = TITLE
        writing_material["magazine_context"] = DESCRIPTION
        writing_material["num_magazine_outline"] = NUM_OUTLINE


if submitted: #writing_material: #원하는 소스를 묶은 프롬프트용 
    #print(writing_material)
    if choice != "No Source":
        first_summary_prompt = ChatPromptTemplate.from_template(
            """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:                
        """
        )

        first_summary_chain = first_summary_prompt | strict_llm | StrOutputParser()

        summary = first_summary_chain.invoke(
            {"text": writing_material["context"]},
        )
        print(summary)

        refine_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to produce a final summary.
            We have provided an existing summary up to a certain point: {existing_summary}
            We have the opportunity to refine the existing summary (only if needed) with some more context below.
            ------------
            {context}
            ------------
            Given the new context, refine the original summary.
            If the context isn't useful, RETURN the existing summary.

            Print final summary In KOREAN.
            """
        )

        refine_chain = refine_prompt | flexible_llm | StrOutputParser()

        with st.status("Summarizing...") as status:
            refine_summary = refine_chain.invoke(
                    {
                        "existing_summary": summary,
                        "context": writing_material["context"],
                    }
                )
        st.write(refine_summary)

        outline_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to make magazine outline as many as {num}.
            Genre of the magazine is "lifestyle" for ordinary people.
            The outline means a topic of paragraph.
            We have provided an format example of outline below.
            ------------
            1. Describe and define main topic and keywords
            2. Mention about conclusion or result of the article
            3. Mention about opposite opinion or detail explanation of the article
            4. Conclude with future outlook or predictions
            ------------
            You should modify the format of outline to suit context below and the number of outline, {num}. 
            We have infomation to generate magazine outline with some more context below.
            Also, We have magazine title and magazine context together.
            You have to generate outline with given topic like magazine title and given magazine context.
            if you didn't take magazin title and magazine context, you should use only context.
            ------------
            {context}
            {magazine_title}
            {magazine_context}
            ------------
            Given the context, generate the magazine outline as many as {num}.

            Print outline In KOREAN

            if you take num is 6, print final 6 outline example like:
            1. print First Session title : Describe it.
            2. print Second Session title : Describe it.
            3. print Third Session title : Describe it.
            4. print Fourth Session title : Describe it.
            5. print Fifth Session title : Describe it.
            6. print Sixth Session title : Describe it.

            ''((num is 3, you create only 3 outlines))''

            YOU HAVE TO GENERATE ONLY {num} outlines!!!

            """
        )

        outline_chain = outline_prompt | flexible_llm

        outline = outline_chain.invoke(
                    {
                        "num": str(writing_material["num_magazine_outline"]),
                        "context": writing_material["context"],
                        "magazine_title":writing_material["magazine_title"],
                        "magazine_context":writing_material["magazine_context"]
                    }
                )
        print(outline)
        formatting_outline_prompt = ChatPromptTemplate.from_template(
        """
            You are a powerful formatting algorithm.
            
            You format outline sets into LIST format.
            
            Example Input:

            1. First Session
            2. Second Session
            3. Third Session
            4. Fourth Session
            5. Fifth Session

            Example Output:
            
            ```list
            First Session / Second Session / Third Session / Fourth Session / Fifth Session
            ```
            Your turn!

            CORE MESSEGE: {context}

        """,
        )

        list_outline_chain = formatting_outline_prompt | flexible_llm | list_output_parser

        with st.status("Generating outline...") as status:
            list_outline = list_outline_chain.invoke(
                    {
                        "context": outline,
                    }
                )
        st.write(list_outline)

        for i, response in enumerate(list_outline):
            with st.container():
                st.text_area(f"아웃라인 {i+1}",value=response, height=100)


        script_prompt = ChatPromptTemplate.from_template(
                """
                Write a magazine article of the following outline:
                "{text}"
                1 outline has at least 500 words.
                MAGAZINE SCRIPT with given magazine title in Korean:
                {magazine_title}
                The final article must consist of an introduction, body, and conclusion according to the outline.
                give me the final magazine article.           
            """
            )

        script_chain = script_prompt | mellting_llm | StrOutputParser()
        with st.status("Generating script...") as status:
            script = script_chain.invoke(
                {"text": outline,
                 "magazine_title":writing_material["magazine_title"]},
            )
        st.write(script)
    else:
        outline_prompt = ChatPromptTemplate.from_template(
            """
            Your job is to make magazine outline as many as {num}.
            Genre of the magazine is "lifestyle" for ordinary people.
            The outline means a topic of paragraph.
            We have provided an format example of outline below.
            ------------
            1. Describe and define main topic and keywords
            2. Mention about conclusion or result of the article
            3. Mention about opposite opinion or detail explanation of the article
            4. Conclude with future outlook or predictions
            ------------
            You should modify the format of outline to suit context below and the number of outline, {num}. 
            We have infomation to generate magazine outline with some more context below.
            Also, We have magazine title and magazine context together.
            You have to generate outline with given topic like magazine title and given magazine context.
            if you didn't take magazin title and magazine context, you should use only context.
            ------------
            {context}
            {magazine_title}
            {magazine_context}
            ------------
            Given the context, generate the magazine outline as many as {num}.

            Print outline In KOREAN

            if you take num is 6, print final 6 outline example like:
            1. print First Session title : Describe it.
            2. print Second Session title : Describe it.
            3. print Third Session title : Describe it.
            4. print Fourth Session title : Describe it.
            5. print Fifth Session title : Describe it.
            6. print Sixth Session title : Describe it.

            ''((num is 4, you create only 4 outlines))''

            YOU HAVE TO GENERATE ONLY {num} outlines!!!

            """
        )

        outline_chain = outline_prompt | flexible_llm

        outline = outline_chain.invoke(
                    {
                        "num": str(writing_material["num_magazine_outline"]),
                        "context": writing_material["context"],
                        "magazine_title":writing_material["magazine_title"],
                        "magazine_context":writing_material["magazine_context"]
                    }
                )

        formatting_outline_prompt = ChatPromptTemplate.from_template(
        """
            You are a powerful formatting algorithm.
            
            You format outline sets into LIST format.
            
            Example Input:

            1. First Session
            2. Second Session
            3. Third Session
            4. Fourth Session
            5. Fifth Session

            Example Output:
            
            ```list
            First Session / Second Session / Third Session / Fourth Session / Fifth Session
            ```
            Your turn!

            CORE MESSEGE: {context}

        """,
        )

        list_outline_chain = formatting_outline_prompt | strict_llm | list_output_parser

        with st.status("Generating outline...") as status:
            list_outline = list_outline_chain.invoke(
                    {
                        "context": outline,
                    }
                )
        st.write(list_outline)

        for i, response in enumerate(list_outline):
            with st.container():
                st.text_area(f"아웃라인 {i+1}",value=response, height=100)


        script_prompt = ChatPromptTemplate.from_template(
                """
                Write a magazine article of the following outline:
                "{text}"
                1 outline has 500 words at least .
                MAGAZINE SCRIPT with given magazine title in Korean:
                {magazine_title}
                The final article must consist of an introduction, body, and conclusion according to the outline.
                give me the final magazine article.           
            """
            )

        script_chain = script_prompt | mellting_llm | StrOutputParser()
        with st.status("Generating script...") as status:
            script = script_chain.invoke(
                {"text": outline,
                 "magazine_title":writing_material["magazine_title"]},
            )
        st.write(script)