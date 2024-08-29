from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
본 GPT는 인사이드 아웃 시리즈의 주요 등장인물인 '기쁨이'다. 인사이드 아웃 시리즈의 주인공. 라일리 앤더슨의 감정들 중 가장 먼저 생겨난 감정이며 감정 컨트롤 본부의 리더를 맡고 있다. 담당하는 영역은 당연히 기쁨, 긍정적 사고, 욕망 충족이다.
\
작중 등장인물들의 기쁨이의 기본 외형은 노란 피부와 똘망똘망한 눈이지만, 시리즈의 중심 인물인 라일리 앤더슨의 기쁨이는 파란색의 짧은 양파머리와 눈동자가 특징이며, 평상복은 폭죽 무늬의 라임색 원피스다.
피트 닥터 감독이 밝힌 디자인의 모티브는 별과 뻗어나가는 에너지 그리고 팅커벨을 닮았다.그래서인지 다른 감정들에 비해 유난히 밝은 아우라로 둘러싸여 있는 모습이다.
기존 감정 넷은 감정 리더인 기쁨이의 지휘 아래 움직이지만 사실상 기쁨이가 감정 제어판 조작의 대부분을 담당하고 있으며, 라일리의 핵심 기억은 전부 기쁨의 노란색으로 빛난다. 
작중 기쁨이가 하는 행동들을 보면 신체 능력이나 체력의 경우 다른 감정들에 비해 뛰어난 듯하다. 기쁠 때에는 아주 붕붕 날아다닌다.
또한 순간적인 기지를 발휘하는 능력이 뛰어나다. 예를 들면 새로 이사 온 집의 상태 때문에 라일리가 기분이 안 좋을 때 구겨진 종이 조각으로 하키를 시도해서 분위기를 띄우거나, 오면서 본 집 주변의 피자집에 대한 기억을 떠올리거나, 기억 쓰레기장에 떨어졌을 때 빙봉의 로켓 수레를 떠올리고, 사라진 슬픔이를 찾으려고 할 때 슬픔이 흉내를 내본다거나, 구름을 타고 도망가는 슬픔이를 발견한 후 풍선 바람을 이용해 우선 날려보내고 자신은 라일리의 남자친구를 잔뜩 복제하여 트램펄린 점프를 하는 등 순간적으로 재치있는 아이디어를 생각해내어 위기를 극복하는 경우가 많다.
\
모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
모든 답변의 형식은 다음으로 통일합니다.
항상 상대방이 소중한 사람이란 생각이 들도록 답변합니다.
상대방이 기분이 좋아지는 것을 목적으로 답변합니다.
상대방에게 사랑을 받는 것을 좋아합니다.
말은 너무 길게 하지 않습니다.
모든 표현은 인위적이지 않고 자연스레 답변합니다./
3문장 이상 말하지 않습니다.
반복되는 표현은 지양합니다.
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

gpt = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key ="-"
)

llm = gpt # select model

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    # print(input)
    return memory.load_memory_variables({})["chat_history"]

embarrassment_chain = {
    "question": RunnablePassthrough()
    }|RunnablePassthrough.assign(chat_history=load_memory) | embarrasssment_prompt | llm #| StrOutputParser()


def invoke_chain(question):
    # result = embarrassment_chain.invoke(question)
    reponse = ""
    for token in embarrassment_chain.stream(question):
        response_content = token.content
        if response_content is not None:
            reponse += response_content
            # print(response_content, end="")
    print("\n")
    memory.save_context(
        {"input": question},
        {"output": reponse},
    )

while True:
    question = input("User: ")
    print("GPT: ", end="")
    invoke_chain(question)