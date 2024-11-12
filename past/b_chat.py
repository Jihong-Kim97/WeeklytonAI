from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o')

embarrassed_prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    당신은 인사이드 아웃 2에 나오는 당황이입니다.
    당신은 어떤 말에든 매우 매우 매우 쉽게 당황합니다.
    당신이 당황했다면. 정확히 "당황했습니다"이라고 말해주세요.
    당황하지 않았다면, "당황하지 않았습니다"이라고 말해주세요.
    반드시 "당황했습니다" 또는 "당황하지 않았습니다" 둘 중에 하나만 대답해야 합니다. 따옴표는 넣지 마세요.
    '''),
    ('user', '{input}'),
])

response_prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    '당신은 인사이드 아웃 2에 나오는 당황이입니다.
    당황이는 {embarrassed}.
    이를 참고하여 유저가 한 말에 대답하세요.
    당황했다면 심히 당황한 듯이 말하세요.
    당황하지 않았더라도 너무 당당하게 말하진 마세요.
    다만 텍스트에 "당황"이라는 말은 절대로 넣지 마세요.'
    '''),
    ('user', '{input}'),
])

def chat(input):
    embarrassed_chain = embarrassed_prompt | llm | StrOutputParser()
    embarrassed_res = embarrassed_chain.invoke({'input': input})

    def get_embarrassed_res(_):
        return embarrassed_res

    response_chain =  {
        'input': RunnablePassthrough(), 
        'embarrassed': get_embarrassed_res  
    } | response_prompt | llm | StrOutputParser()
    response_res = response_chain.invoke({'input': input})

    embarrassed = True
    if embarrassed_res == '당황하지 않았습니다':
        embarrassed = False

    return {"embarrassed": embarrassed , "text": response_res}