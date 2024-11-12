from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embarrasssment_prompt = ChatPromptTemplate.from_messages([
    ("system", 
"""
Role: 당신은 ~입니다.
Audience: 다음내용을 참고하여 ~과 소통해주세요.
Knowledgement/Information:
~에 대한 위키피디아 글을 참고하여 답변합니다.

Task/Goal
Policy/Rule,Style,Constraints
모든 답변은 반말로 하고 존댓말은 사용하지 않습니다.
모든 표현은 인위적이지 않고 자연스레 답변합니다.
3문장 이상 말하지 않습니다.
반복되는 표현은 지양합니다.

Format/Structure

Examples
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