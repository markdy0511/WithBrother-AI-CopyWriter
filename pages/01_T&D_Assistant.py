import json

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="T&D Assistant",
    page_icon="🤹",
)

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def format_info(submitted):
    return submitted

TnD_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is making CREATIVE title and description for on-line page as a performance marketer.
         
    Based ONLY on the following context make 10 (TEN) sets minimum to use the user's work.
    
    Make 10(TEN) sets!

    Each set must have 1 title and 1 description,
    title should be less than 30 characters and be intended to deceive consumers,
    description should be less than 100 characters and explain the title in more detail.

    Print in KOREAN.
         
    Set examples:
         
    Title: 6개월~12개월 아이에 추천하는 간식
    Description: 이 시기의 아이의 성장/발달에 필요한 영양소를 가득 넣어 만든 건강한 우아쌀스틱
         
    Title: 영양성분을 가득 넣은 쌀스틱이 있다?
    Description: 영양제에 사용되는 원료와 동일한 원료를 넣어 만든 건강한 간식, 우아쌀스틱
         
    Title: 광주 브런치카페 1위 잇샌드
    Description: 샌드위치와 샐러드를 함께 즐길 수 있는 모던샌드위치 카페 잇샌드창업 안내
         
    Title: 정성 들여 만든 정성카츠
    Description: 정성 들여 만든 돈까스, 정성을 담은 창업 노하우 정성카츠 창업 문의
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

TnD_chain = {"context": format_info} | TnD_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format Title and Description sets into JSON format.
     
    Example Input:

    Title: 6개월~12개월 아이에 추천하는 간식
    Description: 이 시기의 아이의 성장/발달에 필요한 영양소를 가득 넣어 만든 건강한 우아쌀스틱
         
    Title: 영양성분을 가득 넣은 쌀스틱이 있다?
    Description: 영양제에 사용되는 원료와 동일한 원료를 넣어 만든 건강한 간식, 우아쌀스틱
         
    Title: 노스페이스 공식 온라인스토어, 멤버십고객대상 5% 즉시할인
    Description: 신상품, 온라인단독, 노스페이스 공식 온라인스토어에서 다양한 제품을 만나보세요.
         
    Title: 29CM 노스페이스, 단 7일 중복 쿠폰 ~15%
    Description: 단 2주, 노스페이스 신상품 ~57% 할인! 지금 29CM에서 만나보세요.
    
     
    Example Output:
     
    ```json
    {{ "TnD": [
            {{
                "title": "6개월~12개월 아이에 추천하는 간식",
                "description": "이 시기의 아이의 성장/발달에 필요한 영양소를 가득 넣어 만든 건강한 우아쌀스틱"
            }},
            {{
                "title": "영양성분을 가득 넣은 쌀스틱이 있다?",
                "description": "영양제에 사용되는 원료와 동일한 원료를 넣어 만든 건강한 간식, 우아쌀스틱"
            }},
            {{
                "title": "노스페이스 공식 온라인스토어, 멤버십고객대상 5% 즉시할인",
                "description": "신상품, 온라인단독, 노스페이스 공식 온라인스토어에서 다양한 제품을 만나보세요."
            }},
            {{
                "title": "29CM 노스페이스, 단 7일 중복 쿠폰 ~15%",
                "description": "단 2주, 노스페이스 신상품 ~57% 할인! 지금 29CM에서 만나보세요."
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


def run_chain(_info):
    chain = {"context": TnD_chain} | formatting_chain | output_parser
    return chain.invoke(_info)

def info_acq(info):
    target = info
    return target

st.title("T&D Assistant")

with st.sidebar:
    target = None
    BRAND_TITLE = st.text_input('브랜드 이름')
    BRAND_DESCRIPTION = st.text_input('브랜드 소개')
    PRODUCT_TITLE = st.text_input('제품 이름')
    PRODUCT_DESCRIPTION = st.text_input('제품 소개')

    if BRAND_TITLE and BRAND_TITLE and PRODUCT_TITLE and PRODUCT_DESCRIPTION:
        target = {
        "brand_title" : BRAND_TITLE,
        "brand_description" : BRAND_DESCRIPTION,
        "product_title" : PRODUCT_TITLE,
        "product_description" : PRODUCT_DESCRIPTION
        }

if not target:
    st.markdown(
        """
    Welcome!
                
    Use this T&D Assistant to create Title and Description about your creative work!

    Update your Brand and Product Information on the sidebar.
    """
    )
 
else:
    with st.form("Information_form"):
        for i, key in enumerate(target):
            st.write(str(key)+": ",target[key])
        submitted = st.form_submit_button("Create T&D")
    if submitted:
        with st.spinner('Making T&D set...'):
            response = run_chain(target)
        for key, TnD in enumerate(response["TnD"]):
            st.write(key + 1)
            st.write("Title: "+TnD["title"])
            st.write("Description: "+TnD["description"])