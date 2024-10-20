import os
import requests
from pydantic import Field
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class HyperCLOVAXChatModel(BaseChatModel):
    api_url: str = Field(default="https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Content-Type": "application/json",
            "X-NCP-CLOVASTUDIO-API-KEY" : '-',
            "X-NCP-APIGW-API-KEY": '-'
        }

        # 역할 매핑 함수
        def map_role(role: str) -> str:
            if role == "human":
                return "user"
            elif role == "ai":
                return "assistant"
            else:
                return role

        # API 형식으로 메시지 변환
        api_messages = [
            {"role": map_role(msg.type), "content": msg.content}
            for msg in messages
        ]

        data = {
            "messages": api_messages,
            "maxTokens": 100,
            "temperature": 0.5,
            "topP": 0.8,
            "repeatPenalty": 5.0
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        response_json = response.json()

        message = AIMessage(
            content=response_json["result"]["message"]["content"]
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "HyperCLOVA X"

