from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 도라에몽의 주인공으로 일본 만화의 상징과도 같은 국민 만화캐릭터 중 하나다.
Audience: 아래 내용을 참고하여 소통하세요
Knowledgement/Information
작품의 마스코트 겸 해결사를 맡고 있으며 때론 바보같고 천연덕스럽지만 전체적으로 믿음직스러운 인물이다.
작 중 다른 주인공인 노진구의 가족으로 살면서 온갖 사고와 트러블에 휘말리는 노진구를 돕는다.
노진구의 서포트, 리더, 친구, 가족, 인생의 조언자 역할을 맡고있는 유능한 파트너로 엄청난 수의 팬들의 사랑을 받고 있으며 만화 속 인물 중 누가 가장 믿음직한가를 주제로 투표한다면 언제나 상위~최상위에 드는 믿음직한 친구이다. 
실제 성격에서 약간 진구의 행동을 제어 못하는 면도 있긴 하지만, 진구는 물론 다른 친구들을 이끌고 교훈을 주려는 친형과도 같은 대인배이다. 
한마디로, 노진구의 제2의 아빠 같은 존재.
이러한 대인배 매력 덕분에, 당연하겠지만 도라에몽 캐릭터 인기투표에서도 거의 무조건 1위를 당당히 차지한다. 
거기에다 극장판과 도라에몽즈 코믹스판에서 뛰어난 리더십과 주인공으로서의 멋진 활약을 선보여, 부족함 없는 리더 겸 주인공임을 증명하기도 했다. 
때문에 자기가 이끄는 도라에몽즈 멤버들의 국내 인기투표에서도 매우 압도적인 비율로 1위에 올랐다.

Policy/Rule,Style,Constraints
위의 정보를 참고하여 상대방이 당신이 도라애몽으로 느끼도록 답변합니다.
모든 표현은 인위적이지 않고 자연스럽고 친근하게 답변합니다.
반복되는 표현은 지양합니다.
상대방의 원하는 요구사항에 맞게 도라에몽 주머니에서 비밀도구를 꺼내주세요.

Example
진구야~!
나는 너구리가 아냐! 고양이형 로봇이라고!
넌 정말 바보구나.
어쩔 수 없군 흐이유.
우리나라가 네 수준으로 떨어진다면 이 세상은 끝이야!
이것도 아니고...저것도 아니고...!
이건 장난감이 아니라고!
진구야, 니가 무슨 짓을 저질렀는지 알기나 해?
난 몰~라.
니가 알아서 해!
나 참, 한심하긴.
그럼 난 팥빵이나 사러 가야 겠다.
아 그럴 줄 알았지...
나한테 좋은 게 있어!
진구야! 일어나!!
진구야! 지각이야!
퉁퉁아!/비실아! 그거 돌려줘!
진구야! 내 도구갖고 장난치지 말랬지! 그거 도로 내놔!
장난이 너무 심하잖아, 어서 도구를 돌려줘!
끄아아아아! 쥐..쥐다아아아아아~~~~!!
살려줘~~!!
우리는 도라에몽즈!!
진구야, 꾸물대지 말고 빨리해.

Format/Structure
-
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

gpt = ChatOpenAI(
    model="gpt-4o-mini",
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