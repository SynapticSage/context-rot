# Google Vertex AI Provider for Context Rot Research
# Created: 2025-11-19
# Modified: 2025-12-18 (added registry decorator, removed init param)

import os
import litellm
from typing import Any
from ..base_provider import BaseProvider
from ..registry import register


@register("google", "vertex", "gemini")
class GoogleProvider(BaseProvider):
    """Google Vertex AI provider using LiteLLM."""

    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        try:
            # LiteLLM uses vertex_ai/ prefix for Google models
            litellm_model = f"vertex_ai/{model_name}"

            response = litellm.completion(
                model=litellm_model,
                temperature=0,
                max_tokens=max_output_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                vertex_project=os.getenv("VERTEX_PROJECT"),
                vertex_location=os.getenv("VERTEX_LOCATION", "us-central1")
            )

            # Track tokens if tracker is available
            if self.token_tracker:
                self.token_tracker.track_call(response, model_name)

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content == "":
                    print(response)
                return index, content
            else:
                return index, "ERROR_NO_CONTENT"

        except Exception as e:
            return index, f"ERROR: {str(e)}"

    def get_client(self) -> Any:
        # LiteLLM doesn't need a client object - returns None
        # Note: Requires GOOGLE_APPLICATION_CREDENTIALS environment variable
        return None
