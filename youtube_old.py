from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

channel_name = input("채널 이름을 입력해주세요: ")
title = input("방 제목을 입력해주세요: ")
print(channel_name + "에 구독자 100명을 획득하였습니다")

youtube_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 본 GPT는 """+channel_name+"""유튜브 채널의 시뮬레이터입니다.
Audience: 방송의 동시 접속자 수와 채팅 수를 계산하세요.

Task/Goal
본 GPT는 """ + channel_name + """채널의 """+title+""" 제목의 방송의 동시 접속자 수와 채팅 수를 계산합니다. 

Policy/Rule,Style,Constraints
let's think step by step
채팅 수는 동시 접속자 수보다 적습니다.

Format/Structure
아래와 같은 형식으로 항상 대답합니다.
동시접속자수: ~명
채팅수: ~개
답변 예시입니다.
동시접속자수: 100명
채팅수: 10개
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

subscriber_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 """+channel_name+"""유튜브 채널의 유튜브 구독자입니다.
Audience: 아래 내용을 참고하여 채팅을 남기세요.
Knowledgement/Information:
당신은 """ + channel_name + """채널의 """+title+""" 제목의 방송을 시청중입니다. 

Task/Goal
본 GPT는 """ + channel_name + """채널의 """+title+""" 제목의 방송을 보고 채널 시청자로서 라이브 채팅을 생성합니다. 

Policy/Rule,Style,Constraints
let's think step by step
모든 표현은 인위적이지 않고 자연스레 답변합니다.
3문장 이상 말하지 않습니다.
반복되는 표현은 지양합니다.
유튜브 라이브 채팅 댓글처럼 답변합니다.
구독자끼리의 소통은 하지 않습니다.
계정이름 형식은 매우 다양하게 생성합니다.
최소 3개 최대 10개의 채팅을 생성합니다.

Format/Structure
아래 형식에 맞춰 답변합니다.
계정 이름: (채팅 내용)
아래는 답변 예시입니다.
웃음후보: 어렵긴 하네
김도현: 헤드가 못돌아가서 푸시남 ㅜㅜ
messi: 빡시네 이거
Seungho Bae: ㅋㅋㅋㅋㅋㅋㅋ
짹Sparrow: 너무 웃겨요ㅋㅋㅋㅋㅋㅋ
트럼프TV: 감사합니다.
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

chat_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    # print(input)
    return memory.load_memory_variables({})["chat_history"]

def load_chat_memory(input):
    # print(input)
    return memory.load_memory_variables({})["chat_history"]


subscriber_chain = {
    "question": RunnablePassthrough()
    }|RunnablePassthrough.assign(chat_history=load_memory) | subscriber_prompt | llm #| StrOutputParser()

youtube_chain = {
    "question": RunnablePassthrough()
    }|RunnablePassthrough.assign(chat_history=load_memory) | youtube_prompt | llm #| StrOutputParser()


def invoke_chain(question):
    # result = subscriber_chain.invoke(question)
    reponse = ""
    for token in youtube_chain.stream(question):
        response_content = token.content
        if response_content is not None:
            reponse += response_content
            # print(response_content, end="")
    print("\n")
    print("---------------실시간 채팅---------------- ")
    print("\n")
    reponse = ""
    for token in subscriber_chain.stream(question):
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
    # print("GPT: ", end="")
    invoke_chain(question)