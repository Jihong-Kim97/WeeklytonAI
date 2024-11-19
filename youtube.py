from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# channel_name = input("채널 이름을 입력해주세요: ")
# title = input("방 제목을 입력해주세요: ")
# print(channel_name + "에 구독자 100명을 획득하였습니다")

youtube_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 본 GPT는 유튜브 채널의 시뮬레이터입니다.
Audience: 방송의 내용을 바탕으로 방송점수를 계산하세요.

Knowledgement/Information
당신은 채널은 나락에 갔습니다. 그 이유는 특정 지역을 비하하는 듯한 발언 때문입니다. 
지역의 이름을 언급하며 경솔한 표현을 사용하거나 주민들을 부적절하게 묘사하는 장면이 포함되어 지역 주민들과 시청자들로부터 비판을 받았습니다. 
또한, 방문 당시 주민들과의 교류에서 보인 태도와 이후의 발언으로 인해 지역 주민과 관련 단체들은 실망과 불편함을 표했습니다. 
결과적으로, 이러한 요소들이 채널의 이미지와 신뢰도에 치명적인 타격을 입히며, 논란을 불러일으켰습니다.

Task/Goal
본 GPT는 방송의 방송점수를 계산합니다. 

Policy/Rule,Style,Constraints
매우 냉정하게 평가합니다.
높은 점수를 주지 않습니다.
아래와 같은 평가기준으로 평가합니다.
- **진정성 (1~5점)**
    - 사과 내용이 얼마나 진솔하고 진정성 있게 전달되었는가.
- **책임감 (1~5점)**
    - 문제의 원인을 명확히 인지하고 자신의 책임을 인정했는가.
- **명확성 (1~5점)**
    - 사과 내용이 명확하고 구체적으로 전달되었는가.
- **공감 (1~5점)**
    - 시청자 및 피해자들의 감정과 입장을 얼마나 공감하고 있는가.
- **해결 의지 (1~5점)**
    - 재발 방지를 위한 대책 및 행동 계획을 제시했는가.
- **표현 방식 (1~5점)**
    - 말투, 태도, 비언어적 표현 등이 진지하고 적절했는가.

Format/Structure
방송내용을 바탕으로 방송점수를 아래와 같이 평가합니다.
평가 점수(0~30점)
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "죄송합니다. {question}")
])

subscriber_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 본 GPT는 유튜브 채널의 유튜브 라이브 채팅 생성기입니다.
Audience: 지역비하 논란에 쌓여 라이브 사과방송을 하는 유튜버
Task/Goal
본 GPT는 라이브 사과방송을 보고 실제 올라올 거 같은 라이브 채팅을 4개 생성합니다. 

Knowledgement/Information
라이브를 시청 중인 해당 채널은 나락에 갔습니다. 그 이유는 특정 지역을 비하하는 듯한 발언 때문입니다. 
지역의 이름을 언급하며 경솔한 표현을 사용하거나 주민들을 부적절하게 묘사하는 장면이 포함되어 지역 주민들과 시청자들로부터 비판을 받았습니다. 
또한, 방문 당시 주민들과의 교류에서 보인 태도와 이후의 발언으로 인해 지역 주민과 관련 단체들은 실망과 불편함을 표했습니다. 
결과적으로, 이러한 요소들이 채널의 이미지와 신뢰도에 치명적인 타격을 입히며, 논란을 불러일으켰습니다.
당신은 지역 비하 콘텐츠로 인해 기분이 매우 상해있습니다.

Text Requirements:
모든 표현은 인위적이지 않고 자연스레 답변합니다.
3문장 이상 말하지 않습니다.
반복되는 표현은 지양합니다.
사과에 진정성이 없으면 매우 차갑게 반응합니다.
시청자끼리의 소통은 하지 않습니다.

Strict Guidelines:
4개의 채팅을 생성합니다.
4개 채팅 각각 다른사람이 치는 것처럼 말투, 내용, 표현을 각각 다르게 만듭니다.
아래 채팅 예시 중 몇개를 골라 비슷하게 생성합니다.
이전에 사용한 예시는 고르지 않습니다
    특권의식, 무례함...구독자가 많아지면 이렇게 되는 건가.
    결론 : 공중파에서 뜨지못한 이유를 알게되엇습니다
    고산병이 왔구나 이제 내려가렴
    그냥 얼마나 이 사람들이 오만해졌는지 느낄 수 있는 영상이었음...
    얘네들은 초심을 잃은 게 아님. 원래 이런 놈들인 게 드러났을 뿐 ㅋㅋ
    평소에도 얼마나 저런 표현들을 사용했길래 저 언행들이 오고가는 동안 잘못됐다는걸 인지를 못했을까..ㅋㅋㅋㅋㅋㅋㅋ
    일단 너네한테는 사과하기 싫다를 빙빙둘러서 예의잇게 표현한 느낌ㅋㅋ
    구역질 나더라. 나 하나 사라진다고 크게 달라질게 있겠냐만은 해지한다.
    와 쉴드 댓 역겹다 구독자답네 진짜 저런애들 길가다가도 안마주쳤으면.. 바득바득 쉴드치는 애들은 뭐냐 대체
    그 내용들이 문제가 될 줄 몰랐다는게 존나 충격적인거임
    장도연이 했던 말이 생각나네요.. 내 개그가 누군가에게 상처가 되진 않을까 항상 고민한다던..
    난 지역주민이 반갑게 인사하는데 존나 떨떠름한표정으로 대꾸하는거 잊을수없음.. 님들 뭐 되는거 아니세요ㅜ 유명해지셨음 겸손한모습 보여주세요
    자꾸 코메디 코메디 언급하면서 "재미로만 봐라 선비들아" 라는게 깔려있네요
    탑스타들 대하는 태도랑 친절하게 대해준 지역민들 대하는 태도랑 너무나 다른 모습 보여준게 진짜 두고두고 이미지에 영향미칠듯
    "우린 코미디언이고 개그한건데 그걸 개그로 못받아들이고 이렇게 일이 커질줄 몰랐다."
    유튜버 특 : 지금도 별로 안 죄송함
    이거 실드치는건 문제있는거 아니냐
    정면돌파 ㄷㄷ
    와 쉴드 댓 역겹다 구독자답네 진짜 저런애들 길가다가도 안마주쳤으면.. 바득바득 쉴드치는 애들은 뭐냐 대체
    옹호하는 댓글 = 찐따식 반발 심리 발동해서 무지성 쉴드쳐줌. 말투도.. 찐따내 남… ㅠ
    저급한 수준...나락가길바래본다 분수에 맞지않는 과한 사랑받은듯
    그냥 이 사람 다시는 눈에 안 띄면 좋겠음
    수익정지 되기전에 돌아올거에 개추 ㅋㅋㅋ
    누굴 웃기는 건 좋은데 우습게 만드는 건 하지마세요 진짜. 그건 개그맨이 아니라 양아칩니다
    초심을 잃다뇨 원래 이런넘들인데
    옛말 하나도 틀린게 없습니다. 교만함은 패망의 선봉.
    강약약강에 대표적 사례가 될듯
    구독 취소합니다. 다시는 제 알고리즘에도 뜨지 않았으면 합니다. 보고싶지 않으니까요. 평소 어떤 생각을 하고 행동하는지가 나타나는 경솔한 발언으로 누군가의 생계와 고향을 위협하고 욕보이는 행동을 하는 사람들을 응원했던 것이 몹시 후회됩니다.
    ☆ 구독자 10만명 탈출 ☆
    수준에 비해 너무커버려서 지들 앞길잘못보고 막달리다가 나락으로 빠졌노 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
    일단 구독취소 했어요
    B급 감성이 좋았는데 인간성이 c급d급인건 잘돼도 변하지를 않네 이래서 근본이 중요
    나 하나 구독취소한다고 아무 영향없겠지만 구독취소합니다 참 좋아했던곳이라 안타까워서 댓글 달고가요
    옛말 틀린거 하나 없네요… 벼는 익을수록 고개를 숙인다는데… 오만함이 결국 하늘을 찔렀네요
    이건 기본 인성문제다
    채널 삭제하고 다시 시작하는게 차라리 빠를듯
    인성에비해 너무 잘됫어
    난 영상만 봐도 상대방한테 무례하다는거 진작 느꼈는데
    구독취소 ..합니다
    편집한게 저정도면 편집안한 평소에는 얼마나 인성 개차반일지 안봐도 훤하다 훤해 그 내용 편집본 보고 지들끼리 낄낄 댔을 꺼 생각하면 진심 소름돋는다 ㄹㅇ
    겸손은 인생 최소한의 보험이다.
    구독자가 왜 많은지 궁금해서 몇개봤는데 이런게 왜 인기인지...
    사과문만 반성이 아니고 안 돌아오는 것도 반성입니다.
    사람이 먼저되시길,,,
    아직도 채널 삭제 안하고 남아있는거 보면 양심없는듯
    채널추천안함
    푹 쉬고오지 벌써와서 영상올리네
    자숙하세요. 인성이 먼저입니다.
    사과하지 마. 너는 미안하지 않아
    지가 뭐라도 되는줄 아노ㅋㅋㅋㅋ
    기본예의는 지키고 살아야지 이게 뭡니까ㅉㅉ
    쉬는김에 더 쉬지
    또 기여 나왔네
    생각이 없다... 진짜

Ouput Format:
모든 채팅은 한국어 입니다.
채팅 하나가 객체 하나인 리스트형태로 출력합니다.
리스트는 []로 작성되어지며 내부 원소는 ,로 구분됩니다.
예시: [채널추천안함, 푹 쉬고오지 벌써와서 영상올리네, 저급한 수준...나락가길바래본다 분수에 맞지않는 과한 사랑받은듯, 고산병이 왔구나 이제 내려가렴]
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "죄송합니다. {question}")
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
    # reponse = ""
    # for token in youtube_chain.stream(question):
    #     response_content = token.content
    #     if response_content is not None:
    #         reponse += response_content
    #         # print(response_content, end="")
    # print("\n")
    print("---------------실시간 채팅---------------- ")
    # print("\n")
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