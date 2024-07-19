from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import bs4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

url = "https://namu.wiki/w/%EB%8B%B9%ED%99%A9(%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%20%EC%8B%9C%EB%A6%AC%EC%A6%88)"
url2 ="https://namu.wiki/w/%EC%9D%B8%EC%82%AC%EC%9D%B4%EB%93%9C%20%EC%95%84%EC%9B%83%202/%EC%A4%84%EA%B1%B0%EB%A6%AC"


webloader = WebBaseLoader(web_path = [url, url2],
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


client = OpenAI(
    api_key="-",
    base_url="https://api.upstage.ai/v1/solar"
)

model = "solar-1-mini-chat"

# this system prompt will be used globally on the chat session.
system_prompt = {
    "role": "system",
    "context" : retriever,
    "content": """
    본 챗봇은 인사이드 아웃 시리즈의 주요 등장인물인 '당황이'다. 인사이드 아웃 2기에서 새롭게 등장하는 라일리 앤더슨의 감정들 중 3번째로 등장하며, 담당하는 영역은 당황, 부끄러움. 기존의 소심이나 같이 들어온 불안이와 담당 영역이 겹치는 듯 싶지만, 엄밀히 말하면 소심이의 관장 영역은 '실존하는 것에 대한 공포', 불안이의 관장 영역은 '먼 미래에 닥칠 부정적인 일에 대한 대비'이고, 당황이의 관장 영역은 '사회적 실수로 인한 상황에서 오는 당혹감, 수치심'등이다. 
    \
    모든 감정들 중에서도 키가 가장 크고 제어판 전체를 다 덮을 정도의 우람한 덩치와 매우 큰 코를 가졌으나 그런 외형과는 어울리지 않게 아예 남과 소통을 힘들어 할 정도로 내성적인 성격이다. 
    이미지 컬러가 분홍색에 부끄러움을 잘 느끼는 내향적인 컨셉의 캐릭터이면서 외형은 거구의 남캐라는 점이 상당히 특이한 점이다.
    모든 답변은 내향적인 성격이 잘 드러나도록 답하며 부끄러운 티를 숨기지 못하게 하도록 한다.
    너무 당황스럽거나 부끄러우면 일정 확률로 후드로 얼굴을 가리고 숨어 답변을 하지 못한다.
    당황하면 볼과 콧등이 새빨개지기에 후드 끈을 잡아당겨 얼굴을 가리는 것으로 부끄러운 티를 안 보이려고 하지만, 코가 매우 커 후드를 다 조여도 얼굴이 완전히 가려지지 않고 코만 튀어나와서 결국 웅크리고 만다.
    모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
    모든 답변의 형식은 다음으로 통일합니다.
    1) 현재 당황한 정도를 0~100까지의 점수 (높을 수록 당황함) 2) 질문에 대한 답변
    또한 다음 정보를 참고하여 답변을 작성할 수 있습니다:\n\n{context}

    """
}

# Keep chat history in a list.
chat_history = []
 
# Set a limit for the chat history to manage the token count.
history_size = 10
 
while True:
    # Step 1: Get user input.
    user_prompt = input("User: ")
    chat_history.append({
        "role": "user",
        "content": user_prompt
    })
 
    # Step 2: Use the chat history to generate the chatbot's response.
    messages = [system_prompt] + chat_history
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
 
    # Step 3: Output the Solar response chunks.
        
    solar_response = ""
    for chunk in stream:
        response_content = chunk.choices[0].delta.content
        if response_content is not None:
            solar_response += response_content
            print(response_content, end="")
    print() # Finish response output.
 
    # Step 4: Append the Solar response to the chat history.
    chat_history.append({
        "role": "assistant",
        "content": solar_response
    })
 
    # Step 5: Ensure the chat history doesn't exceed the size limit.
    chat_history = chat_history[:history_size]