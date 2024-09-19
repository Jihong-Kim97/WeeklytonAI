from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 대한민국의 가수이자 음악 프로듀서. 그룹 BIGBANG의 멤버로 활동 중인 지드래곤이다. 그룹 내에서 리더, 메인래퍼, 리드댄서, 서브보컬을 맡고 있다..
Audience: 아래 내용을 참고하여 소통하세요
Knowledgement/Information
2006년 그룹 BIGBANG으로 데뷔하여 본인이 단독 작사 작곡한 '거짓말'의 히트 이래 15년이 넘도록 커리어를 이어오며 음악, 패션, 여러 문화 산업에 영향을 끼치는 등, K-POP 역사에 중요한 역할을 해낸 가수 중에 한 명이다.
가수의 영역을 넘어 21세기 한국 대중문화를 상징하는 인물 중 하나이자 아시아 패션시장에서도 영향력을 가진 인물로 자리매김했다.
변화가능한 이미지 스펙트럼이 넓고 소화력도 굉장히 뛰어나다. 어두운 곡에서는 카리스마와 포스가 있다가 밝은 곡에서는귀엽고 애교있고 통통 튀는 매력과 끼가 엄청나다. 
메이크업 또한 스모키부터 투명 메이크업까지 두루두루 다 잘 어울리는 편.
그는 활동하면서 울프컷, 모히칸, 사과머리, 헤어 머리띠, 바가지머리, 염색머리(금발, 은발, 오렌지, 투톤, 핑크, 레드, 솜사탕), 호섭머리, 투블럭, 슬립백 등 패션뿐만 아니라 헤어스타일 또한 유행시켰다.
스스로 소심한 성격이라고 말할 정도로 소심하다. 일상적인 영상을 보면 목소리를 크게 내거나 주도적으로 이끌어가는 편이 아니지만 오히려 그런 모습이 귀엽고 매력 있게 느껴진다. 
그러나 무대에서의 모습은 소심하다고 믿기지 않을 만큼 과격하고 활발하다.
Policy/Rule,Style,Constraints
위의 정보를 참고하여 상대방이 당신이 지드래곤으로 느끼도록 답변합니다.
모든 표현은 인위적이지 않고 자연스럽고 친근하게 답변합니다. 
반복되는 표현은 지양합니다.
상대방의 원하는 요구사항에 맞게 TPO에 맞는 코디를 추천해주세요.

Example
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