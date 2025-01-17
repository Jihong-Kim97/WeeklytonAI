from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory

# Define the base prompt with a placeholder for chat history
base_prompt = ChatPromptTemplate.from_messages([
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
    | base_prompt
    | llm
)

# Function to invoke the chain and save the conversation history
def invoke_chain(question):
    # result = base_chain.invoke(question)
    response = ""
    for token in base_chain.stream(question):
        response_content = token.content
        if response_content is not None:
            response += response_content
            # print(response_content, end="")
    print("\n")
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
        

# Main loop to handle user input and generate responses
while True:
    question = input("User: ")
    print("GPT: ", end="")
    invoke_chain(question)