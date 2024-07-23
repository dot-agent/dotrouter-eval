import json
import os
import random
import time
from typing import Optional

import openai
import requests
from tenacity import retry, stop_after_attempt, wait_random

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling


class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class RouterCompletionFn(CompletionFn):
    def __init__(self, llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        self.router_url = os.getenv('ROUTER_URL')
        self.headers = {
            'X-Secret-Key': os.getenv('ROUTER_SECRET_KEY'),
            'Content-Type': 'application/json'
        }

    def call_router(self, prompt: str) -> str:
        data = {"prompt": prompt,
                "stream": True}
        response = self._post_request(data)
        if 'status_code' in response and response['status_code'] != 200:
            raise Exception('Router error: ' + response['detail'])
        print(response)
        return response['response']['content']

    @retry(stop=stop_after_attempt(3))
    def _post_request(self, data):
        response = requests.post(self.router_url, headers=self.headers, json=data, stream=True)
        full_output = ''
        for line in response.iter_lines():
            if line:
                output = json.loads(line.decode('utf-8'))['output']
                full_output += output
        
        return full_output


    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.call_router(prompt)
        record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)


