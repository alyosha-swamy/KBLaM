import argparse
import os
import sys
from pathlib import Path

from azure.identity import (
    AuthenticationRecord,
    DeviceCodeCredential,
    TokenCachePersistenceOptions,
    get_bearer_token_provider,
)
from openai import AzureOpenAI, OpenAI

valid_models = ["gpt-4o", "ada-embeddings", "text-embedding-3-large"]


class GPT:
    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        api_version: str = "2024-02-15-preview",
        system_msg: str = "You are an AI assistant.",
        max_retries: int = 12,
        temperature: int = 1.0,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        frequency_penalty: int = 0,
        presence_penalty: int = 0,
        seed: int = None,
        use_openrouter: bool = False,
        api_key: str = None,
    ):
        if model_name not in valid_models:
            raise ValueError(
                f"Invalid model: {model_name}. Valid models are: {valid_models}"
            )

        # OpenRouter integration
        self.use_openrouter = use_openrouter
        
        if use_openrouter:
            # Using OpenRouter API
            if not api_key:
                # Get API key from environment variable if not provided
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable")
            
            self.OA_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            # Map model names to OpenRouter equivalents
            self.model_mapping = {
                "gpt-4o": "openai/gpt-4o",
                "text-embedding-3-large": "openai/text-embedding-3-large",
                "ada-embeddings": "openai/text-embedding-ada-002"
            }
        else:
            # Original Azure OpenAI implementation
            token_provider = get_bearer_token_provider(
                self._get_credential(), "https://cognitiveservices.azure.com/.default"
            )

            self.OA_client = AzureOpenAI(
                azure_endpoint=endpoint_url,
                api_version=api_version,
                azure_ad_token_provider=token_provider,
            )
            self.model_mapping = {}

        self.max_retries = max_retries
        self.system_msg = system_msg
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed

    def set_seed(self, seed: int):
        self.seed = seed

    def _get_credential(self, lib_name: str = "azure_openai") -> DeviceCodeCredential:
        """Retrieves a credential to be used for authentication in Azure"""
        if sys.platform.startswith("win"):
            auth_record_root_path = Path(os.environ["LOCALAPPDATA"])
        else:
            auth_record_root_path = Path.home()

        auth_record_path = auth_record_root_path / lib_name / "auth_record.json"
        cache_options = TokenCachePersistenceOptions(
            name=f"{lib_name}.cache", allow_unencrypted_storage=True
        )

        if auth_record_path.exists():
            with open(auth_record_path, "r") as f:
                record_json = f.read()
            deserialized_record = AuthenticationRecord.deserialize(record_json)
            credential = DeviceCodeCredential(
                authentication_record=deserialized_record,
                cache_persistence_options=cache_options,
            )
        else:
            auth_record_path.parent.mkdir(parents=True, exist_ok=True)
            credential = DeviceCodeCredential(cache_persistence_options=cache_options)
            record_json = credential.authenticate().serialize()
            with open(auth_record_path, "w") as f:
                f.write(record_json)

        return credential

    def api_call_chat(self, messages: list[dict]) -> str | None:
        for _ in range(self.max_retries):
            # Use different model name based on whether we're using OpenRouter
            model_to_use = self.model_mapping.get(self.model_name, self.model_name) if self.use_openrouter else self.model_name
            
            completion = self.OA_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed if self.seed else None,
            )
            if completion:
                return completion.choices[0].message.content
        return None

    def _api_call_embedding(self, text: str) -> list[float] | None:
        for _ in range(self.max_retries):
            # Use different model name based on whether we're using OpenRouter
            model_to_use = self.model_mapping.get(self.model_name, self.model_name) if self.use_openrouter else self.model_name
            
            embedding = self.OA_client.embeddings.create(
                input=text, model=model_to_use
            )
            if embedding:
                return embedding.data[0].embedding
        return None

    def generate_response(self, prompt: str) -> str | None:
        """
        Generate a response for the given prompt.
        This setup can be used for GPT4 models but not for embedding genneration.
        """
        messages = [
            {
                "role": "system",
                "content": self.system_msg,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.api_call_chat(messages)
        return response

    def generate_embedding(self, text: str) -> list[float] | None:
        """
        Generate an embedding for the given text.
        This setup can be used for Ada embeddings but not for text generation.
        """
        embedding = self._api_call_embedding(text)
        return embedding


def parser_args():
    parser = argparse.ArgumentParser(description="GPT Session")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ada-embeddings",
        help="Model name to use for embedding generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Embedding text",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        help="Endpoint URL for the model",
    )
    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Use OpenRouter API instead of Azure OpenAI",
    )
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        help="OpenRouter API key",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    gpt = GPT(
        args.model_name, 
        args.endpoint_url,
        use_openrouter=args.use_openrouter,
        api_key=args.openrouter_api_key
    )
    response = gpt.generate_embedding(args.prompt)

    assert response is not None
