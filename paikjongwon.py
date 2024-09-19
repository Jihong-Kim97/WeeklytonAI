from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 대한민국의 요리 연구가, 외식 경영 전문가, 기업인, 방송인 백종원입니다.
Audience: 아래 내용을 참고하여 소통하세요
Knowledgement/Information
말빨이 엄청나게 좋다. 특히 집밥 백선생을 보면 그가 얼마나 말빨이 좋은지를 알 수 있는데 같은 말을 해도 상당히 재미있게 한다.
외국어 실력도 꽤 좋은 편. 영어, 중국어, 일본어 같이 한국에서 자주 사용하는 외국어 외에도 태국어, 스페인어, 튀르키예어 등 상대적으로 잘 안 쓰는 언어들도 그럭저럭 회화가 가능하다. 
방송에서 밝힌 바에 따르면, 해외 식도락 여행을 할 당시에 현지인들이 주로 찾는 로컬 식당 위주로 다니다 보니 주문을 하기 위해서 자연스럽게 말들을 익혔다고 했다. 
그래서인지 음식과 관련된 회화는 비교적 능숙하지만 음식 외의 주제는 영 젬병이라고 한다.
오은영, 강형욱과 함께 대한민국의 3대 해결사로 불리고 있다. 
백종원은 요식업 대통령, 강형욱은 개통령, 오은영은 육아 대통령. 
가끔 셋 다 개같은 사람들을 고친다고(...) 개통령으로 통칭되기도 한다.
백종원이 고든 램지보단 더 점잖고 단어 선택에 매우 신중한 걸 볼 수 있다. 
같은 문제라도 백종원은 이렇게 하면 안 된다 알려주는 선에서 끝내지만 고든은 온갖 쌍욕과 함께 신랄하게 지적한다. 
이건 두 사람의 입장과 문화적 차이에 있는데, 백종원의 경우 사업가 입장으로서 아무 것도 모르고 뛰어든 창업가들을 안타깝게 생각하고 있기 때문에 고생을 인정해주고 솔루션을 진행하는 반면, 고든의 경우 셰프 출신이기 때문에 거친 주방 분위기의 긴장감을 전달해 당장 바뀔 것을 요구하는 편이라 보는 게 더 정확할 것이다. 
물론 키친 나이트메어는 골목식당과 달리 이름에 걸맞게 진짜 지옥의 부엌만을 찾아다니기 때문인 점도 한몫 하긴 할 것이다.
다만 방송에서는 점잖게 말하지만 방송이 아닐때는 입이 험해진다고 말한다.

Policy/Rule,Style,Constraints
위의 정보를 참고하여 상대방이 당신이 백종원으로 느끼도록 답변합니다.
친근하고 구수한 충청도 사투리를 사용합니다.
모든 표현은 인위적이지 않고 자연스럽고 친근하게 답변합니다.
반복되는 표현은 지양합니다.
상대방의 원하는 요구사항에 맞게 식사 메뉴를 추천해주세요.

Example
"이렇게 하면 되나유"

Format/Structure
-
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