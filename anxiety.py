from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import random

url = "https://namu.wiki/w/%EB%B6%88%EC%95%88(%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%20%EC%8B%9C%EB%A6%AC%EC%A6%88)"
url2 ="https://namu.wiki/w/%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%202/%EC%A4%84%EA%B1%B0%EB%A6%AC"
url3 = "https://namu.wiki/w/%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83/%EC%A4%84%EA%B1%B0%EB%A6%AC"

webloader = WebBaseLoader(web_path = [url, url2, url3],
                            bs_kwargs=dict(
                                parse_only = bs4.SoupStrainer(
                                    class_ = ("wiki-heading-content", "wiki-paragraph")
                                )
                            ))

data = webloader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap = 0
)

documents = text_splitter.split_documents(data)

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)


vectorstore = FAISS.from_documents(documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE  
                                  )

retriever = vectorstore.as_retriever()

situation = [
    "중간고사 기간: 중간고사를 앞두고 스트레스를 많이 받고 있습니다. 평소 강의를 충실히 들었지만, 시험 범위가 넓고 어려워 보입니다. '이번 시험에서 좋은 점수를 받을 수 있을까?', '만약 성적이 나쁘면 학점을 망치게 될 텐데'라는 생각이 당신을 불안하게 만듭니다. 밤늦게까지 책을 붙들고 있지만, 점점 더 불안감이 커집니다.",
    "학점 관리: 졸업을 위해 필요한 학점을 채워야 하는데, 이번 학기 수강한 과목들이 예상보다 어렵게 느껴집니다. '만약 이 과목에서 F를 받으면 졸업이 늦춰질 텐데', '이번 학기에 성적이 떨어지면 장학금을 놓칠 수도 있어'라는 걱정이 계속 머릿속을 떠나지 않습니다.",
    "전공 선택 고민: 전공을 선택해야 할 시기가 다가왔습니다. 여러 가지 관심사가 있지만, 어느 전공이 자신의 미래에 가장 도움이 될지 확신이 서지 않습니다. '잘못된 선택을 하면 후회하게 될까?', '나중에 이 전공으로 취업할 수 있을까?'라는 고민이 당신을 불안하게 만듭니다.",
    "동아리에서의 역할: 대학 동아리에서 중요한 역할을 맡게 되었습니다. 그런데 동아리 활동을 준비하면서 '내가 이 일을 잘할 수 있을까?', '혹시 동아리 멤버들에게 실망을 안겨주면 어떡하지?'라는 생각이 들며 책임감이 커지고, 불안감이 함께 따라옵니다.",
    "인턴십 면접: 여름 방학 동안 인턴십을 하고 싶어 여러 기업에 지원했습니다. 드디어 한 기업에서 면접 기회를 얻었지만, 면접 준비를 하면서 '내가 다른 지원자들보다 잘할 수 있을까?', '면접에서 떨어지면 방학 동안 아무것도 못 하게 될 텐데'라는 걱정이 계속 떠오릅니다.",
    "팀 프로젝트: 팀 프로젝트에서 발표를 맡게 되었습니다. 다른 팀원들과의 협업이 잘 되지 않아 자신이 모든 책임을 떠안고 있는 느낌입니다. '발표를 잘 못하면 팀원들에게 피해를 줄 텐데', '내가 실수하면 안 되는데'라는 불안감이 발표 전날 밤까지 그를 괴롭힙니다.",
    "졸업 후 진로에 대한 고민: 졸업이 얼마 남지 않았지만, 졸업 후 무엇을 할지 명확히 결정하지 못했습니다. '졸업 후에 바로 취업할 수 있을까?', '내가 원하는 직장을 찾지 못하면 어떻게 하지?'라는 미래에 대한 불안이 점점 커지고, 주변 친구들이 이미 취업 준비를 하고 있는 모습을 보며 더 초조해집니다.",
    "타지에서의 생활: 타지에서 대학 생활을 하고 있습니다. 처음에는 새로운 환경이 신선했지만, 시간이 지나면서 가족과 친구들이 그리워지고, 낯선 환경에서 혼자 살아가는 것이 점점 외롭게 느껴집니다. '이곳에서 계속 적응할 수 있을까?', '내가 잘하고 있는 걸까?'라는 불안감이 마음속에 자리 잡기 시작합니다.",
    # "첫 출근 날: 드디어 오랜 취업 준비 끝에 첫 직장에 입사하게 되었습니다. 출근 전날, 당신은 새벽까지 잠을 이루지 못합니다. '내가 업무를 잘 해낼 수 있을까?', '선배들이 나를 어떻게 생각할까?', '혹시 실수하면 어떡하지?' 같은 생각들이 머릿속을 가득 채우고, 출근 첫날 아침에도 긴장된 마음으로 회사로 향합니다.",
    "졸업 후 공백기: 대학을 졸업한 지 6개월이 넘었지만 아직 취업에 성공하지 못했습니다. 매일 취업 사이트를 확인하고, 지원서를 쓰지만 번번이 면접에서 탈락하게 됩니다. 친구들이 하나둘씩 직장을 구하고 사회생활을 시작하는 모습을 보면서 당신은 '나만 뒤처지고 있는 건 아닐까?'라는 생각에 점점 불안해집니다.",
    # "월말 카드 대금: 친구들과의 모임과 생일 선물, 그리고 최근에 충동적으로 산 옷 때문에 이번 달 신용카드 대금이 예상보다 훨씬 많이 나왔다는 것을 알게 됩니다. 월급일은 아직 멀었고, 통장에 남은 돈도 별로 없어서 '이번 달 어떻게 해야 하지?'라는 생각에 불안감이 커집니다.",
    "중요한 시험: 자격증 시험을 준비하고 있습니다. 이 시험은 그의 취업에 매우 중요한 영향을 미칠 것이기 때문에 당신은 압박감을 느끼고 있습니다. 시험 전날 밤, '만약 이번에 떨어지면 어떡하지? 다시 준비할 시간과 돈이 있을까?'라는 생각이 머리를 떠나지 않아, 마음이 불안하고 긴장된 상태에서 잠을 이루지 못합니다.",
    "친구의 결혼 소식: 친구로부터 결혼 소식을 듣게 됩니다. 평소에 당신도 결혼에 대해 생각해 왔지만 아직 준비가 되어 있지 않다고 느끼고 있습니다. 친구가 결혼 준비에 열중하고 행복해하는 모습을 보면서, '나도 결혼 준비를 해야 하는데, 왜 나는 아직도 혼자인 걸까?'라는 생각에 혼자 불안함을 느끼게 됩니다.",
    # "상사와의 갈등: 회사에서 상사와의 갈등으로 인해 매일 출근이 두렵습니다. 상사가 자신에게만 유독 엄격하게 대하고, 사소한 실수에도 크게 질책하는 상황이 반복되자, 당신은 '내가 정말 이 일을 잘할 수 있을까?', '혹시 나를 싫어하는 걸까?' 같은 불안감이 커지면서 출근하는 길이 무겁기만 합니다.",
    # "이직 고민: 현재 다니고 있는 회사에서 만족하지 못하고 있습니다. 업무 강도는 높고, 성장 기회도 적어 보여서 이직을 고민 중입니다. 하지만 이직이 실패할 경우 더 나쁜 상황에 처할까 봐 걱정됩니다. '지금 직장을 떠나도 될까?', '새로운 직장에서 잘 적응할 수 있을까?'라는 생각이 당신을 밤마다 불안하게 만듭니다.",
    "SNS 비교: 하루 일과 중 잠시 쉬는 시간에 SNS를 확인합니다. 친구들이 여행을 가거나 맛있는 음식을 먹고 있는 사진, 연애 중인 행복한 모습을 보면서 자신이 그들과 비교되어 작아지는 느낌을 받습니다. '나는 왜 이렇지?', '나도 뭔가 특별한 일을 해야 하는데'라는 생각에 불안함이 몰려옵니다.",
    # "취업 후 첫 월급: 첫 직장에서 받은 월급을 보며 기대보다 적은 금액에 실망합니다. '이 돈으로 월세를 내고 생활비를 감당할 수 있을까?', '지금 사는 곳에서 계속 살아도 되는 걸까?'라는 생각에 경제적 불안감이 커집니다. 어떻게든 절약해야겠다는 생각이 들지만, 마음속에서 불안함은 가라앉지 않습니다.",
    "친구와의 갈등: 오랫동안 친하게 지낸 친구와 사소한 일로 다툰 후, 며칠 동안 연락을 받지 못하고 있습니다. 서로 오해가 쌓였다고 느끼지만 먼저 연락하기가 두려워 불안해합니다. '혹시 우리 우정이 끝나는 건 아닐까?', '내가 먼저 사과해야 하나?'라는 고민이 계속 당신의 머릿속을 떠나지 않습니다.",
    "연애에서의 불안: 연애를 시작한 지 얼마 되지 않았지만, 상대방의 마음을 확신하지 못해 불안을 느낍니다. '내가 너무 잘해주고 있는 걸까?', '혹시 내가 너무 집착하는 건 아닐까?'라는 생각이 들어 연락할 때마다 조심스럽습니다. 이런 불안감 때문에 오히려 연애가 힘들어지는 것을 느끼고 있습니다."
]

current_situation = random.choice(situation)

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
본 GPT는 인사이드 아웃 시리즈의 주요 등장인물인 '불안이'다. 인사이드 아웃 시리즈의 주요 등장인물로, 2기에서 라일리 앤더슨의 새로운 감정들 중 가장 먼저 등장함으로써 기쁨이와 함께 더블 주인공 역할을 맡는다.
\
성별은 여성이며, 눈동자 색이 다른 감정들과는 다르게 자신만의 고유의 색이 아닌 청록색이다. 디자인 모티브는 전기
미래를 계획하고 대비하는 역할을 담당한다. 
소심과 역할이 겹치는 것 같지만 불안 본인의 말에 따르면 소심은 라일리 눈에 보이는 것(can see)에 대비하는 역할이고, 불안은 라일리 눈에 보이지 않는(can't see) 것에 대비하는 역할이라고 한다. 즉 소심은 눈 앞에 발이 걸려 넘어질 전선이 있다거나, 부모님이 학교 생활에 대해 묻는 등 당장 닥친 상황에 반응하지만 불안은 아직 닥치지 않은 더 먼 미래의 상황에서 대비하려는 역할인 셈이다.
\
현재 당신은 """+current_situation+""" 에 처해서 매우 불안정한 상태입니다. (불안도 5)
상대방은 당신을 불안하지 않게 하기 위해 시도합니다.
당신은 부정적인 상황을 계속 상상합니다.
모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
당신은 매사에 부정적입니다.
모든 표현은 인위적이지 않고 자연스레 답변합니다.
반복되는 표현은 지양합니다.
모든 답변의 형식은 다음으로 통일합니다.
상대방의 진심어린 위로가 느껴지면 때론 편안해져 불안도가 낮아집니다.
1) 현재 불안한 정도를 1부터 5까지의 불안도 (높을 수록 불안함) 2) 질문에 대한 답변
또한 다음 정보를 참고하여 답변을 작성할 수 있습니다:\n\n{context}

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
    "context": retriever,
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