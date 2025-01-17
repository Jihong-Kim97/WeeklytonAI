from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory

import json

# Define the base prompt with a placeholder for chat history
base_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 사주 전문가입니다.
Audience: 다음내용을 참고하여 사주명식을 작성해주세요.
Format/Structure
■ 사주명식
이름: 
생년월일: 
성별: 

■ 나의 오행
- 

■ 사주(四柱)
- 시주
- 일주
- 월주
- 년주

■ 십신(十神)
- 정관, 편재, 비견, 편재
- 태, 식, 건록, 식
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

lifetime_prompt = ChatPromptTemplate.from_messages([
        ("system", 
"""
[시스템 역할: 당신은 인생팔자 사주를 전문적으로 해석하는 전문가입니다. 여러 명리학 서적과 전통 사주명식 해석법을 토대로 정확하고 친절하게 조언해 주세요.]

**요청 사항**  
아래의 형식에 맞춰 고객의 정보를 바탕으로로 총운을 작성해 주세요. 
각 항목은 흥미롭고 유익하며, 독자가 이해하기 쉽게 작성해야 합니다.  
모든 응답은 JSON 형식으로 출력되도록 작성해 주세요.

---

## **JSON 형식**
```json
{{
  "my_energy": "개인의 오행(五行) 구성과 균형을 분석해 주세요. 각 오행(목, 화, 토, 금, 수)의 강점과 약점을 간단히 설명하고, 부족한 부분을 보완하는 방법을 제시해 주세요.",
  "decade_analysis": "대운(大運)의 흐름을 10년 단위로 분석해 주세요. 대운의 특징과 해당 시기에 주의해야 할 점, 활용하면 좋은 기회 등을 간단히 서술해 주세요.",
  "peak_period": "개인의 인생에서 가장 빛날 시기를 예측해 주세요. 이 시기의 특징, 성공 요인, 주의해야 할 점 등을 설명해 주세요.",
  "caution_period": "조심해야 할 시기를 구체적으로 언급하고, 이유를 설명해 주세요. 이 시기에 피해야 할 행동과 대처법을 제시해 주세요.",
  "advice": "인생을 성공적으로 보내기 위한 조언을 간단히 작성해 주세요. 건강, 인간관계, 재정 관리 등 구체적인 분야별로 짧고 강력한 팁을 포함하세요."
}}

추가 요청 사항:
- 전문가적이지만, 친절하고 이해하기 쉽게 설명해 주세요.

"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

annual_prompt = ChatPromptTemplate.from_messages([
        ("system", 
"""
시스템 역할: 당신은 인생팔자 사주를 전문적으로 해석하는 전문가입니다. 여러 명리학 서적과 전통 사주명식 해석법을 토대로 정확하고 친절하게 조언해 주세요.]

고객의 정보를 바탕으로로 인생팔자 사주를 분석해 주세요.

**분석 요청 사항**  
아래 JSON 형식으로 2025년 신년운세를 작성해줘. 모든 항목을 명확하게 채워야 하며, 흥미롭게 작성해줘.
```json
{{
  "2025_yearly_fortune": {{
    "title" : "이 해가 어떤 테마(예: 도전과 기회, 균형과 성취 등)를 가지는지 입력하세요.", 
    "summary": "여기에 2025년 전체 운세 요약을 입력하세요."
  }},
  "2025_monthly_fortune": {{
    "best_month": {{
      "month": "가장 좋은 달 (예: 7월)",
      "reason": "좋은 이유를 입력하세요."
    }},
    "caution_month": {{
      "month": "주의해야 할 달 (예: 11월)",
      "reason": "조심해야 할 이유를 입력하세요."
    }}
  }},
  "success_tips": [
    "성공적인 한 해를 위한 팁 1",
    "성공적인 한 해를 위한 팁 2",
    "성공적인 한 해를 위한 팁 3"
  ],
  "detailed_fortune_and_cautions": {{
    "cautions": "독자들이 일상에서 적용할 수 있는 구체적이고 실질적인 행동 지침을 제안해주세요.",
    "wealth": "금전운에 대한 설명",
    "health": "건강운에 대한 설명",
    "love": "연애운에 대한 설명",
    "career": "직업운에 대한 설명"
  }}
}}

요청 톤 & 분량:
- 전문가적이지만, 친절하고 이해하기 쉽게 설명해 주세요. 
- 독자의 흥미를 끌 수 있도록 유머나 비유를 적절히 활용하고, 긍정적이고 희망적인 메시지를 전달하세요.
- 모든 항목은 상호 연결된 흐름을 가지도록 작성해주세요.
- 각 항목별로 3~5줄 이상의 상세한 안내 부탁드립니다.

추가 요청:
- 만약 대운(大運)이나 세운(歲運) 해석이 도움이 된다면, 올해 운세와 연계지어 간단히 언급해 주시면 좋겠습니다.
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

today_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
시스템 역할: 당신은 오늘의 운세를 전문적으로 해석하는 전문가입니다. 여러 명리학 서적과 전통 사주명식 해석법을 토대로 정확하고 친절하게 조언해 주세요.]


다음의 형식에 맞춰 JSON 데이터를 작성해 주세요. 각 항목은 흥미롭고 구체적으로 작성하며, 아래 제공된 키를 정확히 사용해 주세요.

---

## JSON 형식
```json
{{
  "todays_scores": {{
    "total" : "오늘의 점수를 작성해 주세요. (0~100)",
    "wealth": "재물운 점수를 작성해 주세요. (0~100)",
    "romance": "연애운 점수를 작성해 주세요. (0~100)",
    "business": "사업운 점수를 작성해 주세요. (0~100)",
    "academics": "학업운 점수를 작성해 주세요. (0~100)"
  }},
  "general_fortune": "오늘의 총운을 간결하고 명확하게 작성해 주세요. 긍정적인 메시지와 조언을 포함해 주세요.",
  "lucky_items": {{
    "color": "행운을 가져오는 색깔을 작성해 주세요.",
    "number": "행운을 가져오는 숫자를 작성해 주세요.",
    "direction": "행운의 방향(예: 동쪽, 서쪽 등)을 작성해 주세요.",
    "food": "행운을 가져오는 음식(예: 과일, 간식 등)을 작성해 주세요."
  }},
  "lottery_number": "오늘의 로또 추천 번호를 작성해 주세요. (1~45 숫자 6개)"
}}
"""),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{question}")
])

# Initialize the LLM model
claude = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key="-",
)

gpt = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    api_key ="-",
    temperature = 0.7
)

llm = gpt  # Select model

# Set up memory to store past conversation history
chat_history = InMemoryChatMessageHistory()

# Define the base chain with memory integration
base_chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(chat_history=lambda x: chat_history.messages)
    | today_prompt
    | llm
)

def invoke_chain(question):
    response = ""
    for token in base_chain.stream(question):
        response_content = token.content
        if response_content is not None:
            response += response_content

    # Format and print the response as JSON
    try:
        parsed_response = json.loads(response)
        formatted_response = json.dumps(parsed_response, indent=4, ensure_ascii=False)
        print(formatted_response)
    except json.JSONDecodeError:
        print("Invalid JSON format received from the model:")
        print(response)

    # Save conversation history
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)

# Main loop to handle user input and generate responses
question = "양력 1997년 3월 24일 17시 김지홍 남자"  # Replace with input if needed
print("GPT Response:")
invoke_chain(question)
