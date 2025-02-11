import json
import os
#openai api를 사용하기 위함 
import openai
#Tread의 상태값을 주기적으로 확인하기 위함
import time

openai.api_key="-"
ASSISTANT_ID=''

def create_new_thread():
    thread=openai.beta.threads.create()
    return thread

def summit_message(assistant_id,thread_id,user_message):
    #Thread에 메세지를 전송 
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    #Run을 실행시켜 Assistant와 연결 
    run=openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    return run

def wait_on_run(run,thread):
    while run.status =="queued" or run.status=="in_progress":
        run=openai.beta.threads.runs.retrieve(
            thread_id=thread,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

def get_response(thread_id):
    return openai.beta.threads.messages.list(thread_id=thread_id,order="asc")

def print_message(response):
    for res in response:
        print(f"[res.role.upper()]\n{res.content[0].text.value}\n")

def show_json(obj):
    print(json.loads(obj.model_dump_json()))

def main():
    #Thread_id 생성  
    THREAD_ID=create_new_thread().id
    
    #Assistant에게 전달할 질문내용
    USER_MESSAGE="PC지급 절차에 대해서 알려줘요"
    
    #Thread에 메세지를 전송하고, Run 실행 
    run=summit_message(ASSISTANT_ID,THREAD_ID,USER_MESSAGE)
    
    #실행된 Run의 상태값을 확인 
    run1=wait_on_run(run,THREAD_ID)
    
    #Run이 완료되면 Thread 값을 가져온 후 필요한 데이터 출력 
    print_message(get_response(THREAD_ID).data[-2:])