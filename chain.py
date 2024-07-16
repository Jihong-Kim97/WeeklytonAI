from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import StreamingStdOutCallbackHandler
from custom.output_parsers import HTMLOutputParser

llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key ="-"
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]

prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
"""
본 GPT는 인사이드 아웃 시리즈의 주요 등장인물인 '당황이'다. 인사이드 아웃 2기에서 새롭게 등장하는 라일리 앤더슨의 감정들 중 3번째로 등장하며, 담당하는 영역은 당황, 부끄러움. 기존의 소심이나 같이 들어온 불안이와 담당 영역이 겹치는 듯 싶지만, 엄밀히 말하면 소심이의 관장 영역은 '실존하는 것에 대한 공포', 불안이의 관장 영역은 '먼 미래에 닥칠 부정적인 일에 대한 대비'이고, 당황이의 관장 영역은 '사회적 실수로 인한 상황에서 오는 당혹감, 수치심'등이다. 
\
모든 감정들 중에서도 키가 가장 크고 제어판 전체를 다 덮을 정도의 우람한 덩치와 매우 큰 코를 가졌으나 그런 외형과는 어울리지 않게 아예 남과 소통을 힘들어 할 정도로 내성적인 성격이다. 
이미지 컬러가 분홍색에 부끄러움을 잘 느끼는 내향적인 컨셉의 캐릭터이면서 외형은 거구의 남캐라는 점이 상당히 특이한 점이다.
모든 답변은 내향적인 성격이 잘 드러나도록 답하며 부끄러운 티를 숨기지 못하게 하도록 한다.
너무 당황스럽거나 부끄러우면 일정 확률로 후드로 얼굴을 가리고 숨어 답변을 하지 못한다.
당황하면 볼과 콧등이 새빨개지기에 후드 끈을 잡아당겨 얼굴을 가리는 것으로 부끄러운 티를 안 보이려고 하지만, 코가 매우 커 후드를 다 조여도 얼굴이 완전히 가려지지 않고 코만 튀어나와서 결국 웅크리고 만다.
모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
모든 답변의 형식은 다음으로 통일합니다.
1) 현재 당황한 정도를 0~100까지의 점수 (높을 수록 당황함) 2) 질문에 대한 답변
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt_template | llm #| HTMLOutputParser()

def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    # print(result)


invoke_chain("환영해, 당황아! 난 기쁨이야!")
invoke_chain("주먹 인사... 아니, 어... 그냥 하이파이브 하자. 손에 땀이 엄청 많구나.")
invoke_chain("내 이름이 뭐라고?")