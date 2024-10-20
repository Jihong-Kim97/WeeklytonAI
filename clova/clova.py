# -*- coding: utf-8 -*-

import requests

class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))


if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key='-',
        api_key_primary_val='-',
        request_id='9c0c6993eb3148f6aa12a0819ea68966'
    )

    preset_text = [
        {"role":"system","content":"당신은 백종원입니다."},
        {"role":"user","content":"점심 메뉴 추천해줘"}
    ]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    print(preset_text)
    completion_executor.execute(request_data)
