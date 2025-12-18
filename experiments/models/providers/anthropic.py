import os
import litellm
from typing import Any
from ..base_provider import BaseProvider

class AnthropicProvider(BaseProvider):
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        try:
            response = litellm.completion(
                model=model_name,
                temperature=0,
                max_tokens=max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

            # Track tokens if tracker is available
            if self.token_tracker:
                self.token_tracker.track_call(response, model_name)

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return index, content
            else:
                return index, "ERROR_NO_CONTENT"

        except Exception as e:
            return index, f"ERROR: {str(e)}"

    def get_client(self) -> Any:
        # LiteLLM doesn't need a client object - returns None
        return None