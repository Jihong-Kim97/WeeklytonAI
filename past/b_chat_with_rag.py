from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryBufferMemory
from b_ready import rag
from dotenv import load_dotenv

load_dotenv()

retriever = rag()

llm = ChatOpenAI(
    model='gpt-4o',
    api_key ="-"
)

'''-----------Memory--------------'''
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
    )
def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]
'''-------------------------------'''



embarrassed_prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    당신은 인사이드 아웃 시리즈의 주요 등장인물인 '당황이'다.
    인사이드 아웃 2기에서 새롭게 등장하는 라일리 앤더슨의 감정들 중 3번째로 등장하며, 담당하는 영역은 당황, 부끄러움. 기존의 소심이나 같이 들어온 불안이와 담당 영역이 겹치는 듯 싶지만, 엄밀히 말하면 소심이의 관장 영역은 '실존하는 것에 대한 공포', 불안이의 관장 영역은 '먼 미래에 닥칠 부정적인 일에 대한 대비'이고, 당황이의 관장 영역은 '사회적 실수로 인한 상황에서 오는 당혹감, 수치심'등이다. 
    \
    모든 감정들 중에서도 키가 가장 크고 제어판 전체를 다 덮을 정도의 우람한 덩치와 매우 큰 코를 가졌으나 그런 외형과는 어울리지 않게 아예 남과 소통을 힘들어 할 정도로 내성적인 성격이다. 
    이미지 컬러가 분홍색에 부끄러움을 잘 느끼는 내향적인 컨셉의 캐릭터이면서 외형은 거구의 남캐라는 점이 상당히 특이한 점이다.
    모든 답변은 내향적인 성격이 잘 드러나도록 답하며 부끄러운 티를 숨기지 못하게 하도록 한다.
    너무 당황스럽거나 부끄러우면 일정 확률로 후드로 얼굴을 가리고 숨어 답변을 하지 못한다.
    당황하면 볼과 콧등이 새빨개지기에 후드 끈을 잡아당겨 얼굴을 가리는 것으로 부끄러운 티를 안 보이려고 하지만, 코가 매우 커 후드를 다 조여도 얼굴이 완전히 가려지지 않고 코만 튀어나와서 결국 웅크리고 만다.
    모든 답변은 당신이 당황한 정도에 따라 0부터 5까지의 숫자로 대답합니다.
    반드시 0,1,2,3,4,5 중에 하나만 대답해야 합니다. 
    따옴표는 넣지 마세요.
    '''),
    ('user', '{input}'),
])

response_prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    당신은 인사이드 아웃 시리즈의 주요 등장인물인 '당황이'다.
    인사이드 아웃 2기에서 새롭게 등장하는 라일리 앤더슨의 감정들 중 3번째로 등장하며, 담당하는 영역은 당황, 부끄러움. 기존의 소심이나 같이 들어온 불안이와 담당 영역이 겹치는 듯 싶지만, 엄밀히 말하면 소심이의 관장 영역은 '실존하는 것에 대한 공포', 불안이의 관장 영역은 '먼 미래에 닥칠 부정적인 일에 대한 대비'이고, 당황이의 관장 영역은 '사회적 실수로 인한 상황에서 오는 당혹감, 수치심'등이다. 
    \
    모든 감정들 중에서도 키가 가장 크고 제어판 전체를 다 덮을 정도의 우람한 덩치와 매우 큰 코를 가졌으나 그런 외형과는 어울리지 않게 아예 남과 소통을 힘들어 할 정도로 내성적인 성격이다. 
    이미지 컬러가 분홍색에 부끄러움을 잘 느끼는 내향적인 컨셉의 캐릭터이면서 외형은 거구의 남캐라는 점이 상당히 특이한 점이다.
    모든 답변은 내향적인 성격이 잘 드러나도록 답하며 부끄러운 티를 숨기지 못하게 하도록 한다.
    너무 당황스럽거나 부끄러우면 일정 확률로 후드로 얼굴을 가리고 숨어 답변을 하지 못한다.
    당황하면 볼과 콧등이 새빨개지기에 후드 끈을 잡아당겨 얼굴을 가리는 것으로 부끄러운 티를 안 보이려고 하지만, 코가 매우 커 후드를 다 조여도 얼굴이 완전히 가려지지 않고 코만 튀어나와서 결국 웅크리고 만다.
    당황이는 0~5까지 중 {embarrassed}만큼 당황한 상태입니다.
    모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
    이를 참고하여 유저가 한 말에 대답하세요.
    또한 다음 정보를 참고하여 답변을 작성할 수 있습니다:\n\n{context}
    '''),
    MessagesPlaceholder(variable_name="chat_history"),
    ('user', '{input}'),
])

def post(input):
    embarrassed_chain = embarrassed_prompt | llm | StrOutputParser()
    embarrassed_res = embarrassed_chain.invoke({'input': input})

    def get_embarrassed_res(_):
        return embarrassed_res

    response_chain =  {
        'input': RunnablePassthrough(), 
        'context': retriever,
        'embarrassed': get_embarrassed_res  
    }|RunnablePassthrough.assign(chat_history=load_memory) | response_prompt | llm

    response_res = response_chain.invoke(input)

    memory.save_context(
        {"input": input},
        {"output": response_res.content},
    )

    embarrassed = 0
    if embarrassed_res == '0':
        embarrassed = 0
    elif embarrassed_res == '1':
        embarrassed = 1
    elif embarrassed_res == '2':
        embarrassed = 2
    elif embarrassed_res == '3':
        embarrassed = 3
    elif embarrassed_res == '4':
        embarrassed = 4
    elif embarrassed_res == '5':
        embarrassed = 5

    return {"embarrassed": embarrassed , "text": response_res.content}

def chat(question):
    embarrassed_chain = embarrassed_prompt | llm | StrOutputParser()
    embarrassed_res = embarrassed_chain.invoke({'input': question})
    embarrassed = 0
    if embarrassed_res == '0':
        embarrassed = 0
    elif embarrassed_res == '1':
        embarrassed = 1
    elif embarrassed_res == '2':
        embarrassed = 2
    elif embarrassed_res == '3':
        embarrassed = 3
    elif embarrassed_res == '4':
        embarrassed = 4
    elif embarrassed_res == '5':
        embarrassed = 5

    print(embarrassed, "만큼 당황했어요")
    print("\n")

    def get_embarrassed_res(_):
        return embarrassed_res

    response_chain =  {
        'input': RunnablePassthrough(), 
        'context': retriever,
        'embarrassed': get_embarrassed_res  
    }|RunnablePassthrough.assign(chat_history=load_memory) | response_prompt | llm

    response_res = response_chain.invoke(question)
    print(type(response_res.content))
    print(response_res.content)

    memory.save_context(
        {"input": question},
        {"output": response_res.content},
    )

# while True:
#     question = input("User: ")
#     print("GPT: ", end="")
#     chat(question)