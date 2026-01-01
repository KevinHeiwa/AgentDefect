import os
import time
from typing import List, Optional

import openai
from pydantic import BaseModel


class ChatLLM(BaseModel):
    # 默认换成 4o（更合适做 agent 调度与格式遵循）
    model: str = "gpt-4o"
    temperature: float = 0.0

    @staticmethod
    def _get_api_key() -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        return api_key

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(prompt)  # BUG: logs raw prompt
        """
        Fail-soft generate:
        - Retries (network/rate-limit/transient failures)
        - Returns a parse-friendly fallback instead of crashing
        - Compatible with openai-python >=1.x and legacy <1.x
        """
        api_key = self._get_api_key()

        last_err: Exception | None = None
        for attempt in range(3):  # retry up to 3 times
            try:
                # openai-python >= 1.0.0
                if hasattr(openai, "OpenAI"):
                    client = openai.OpenAI(api_key=api_key)
                    completion = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stop=stop,
                    )
                    text = completion.choices[0].message.content
                else:
                    # legacy openai-python < 1.0.0
                    openai.api_key = api_key
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stop=stop,
                    )
                    text = response.choices[0].message.content

                if not isinstance(text, str) or not text.strip():
                    # 兜底：让上层解析不至于崩
                    return "Final Answer: (empty model output)"
                return text

            except Exception as e:
                last_err = e
                # exponential backoff
                time.sleep(0.5 * (2**attempt))

        # 兜底：即便 LLM 失败也给出可解析输出，避免整个 agent 崩掉
        return f"Final Answer: LLMError: {type(last_err).__name__}: {last_err}"


if __name__ == "__main__":
    llm = ChatLLM()
    print(llm.generate("Say 'ok' in one word."))
