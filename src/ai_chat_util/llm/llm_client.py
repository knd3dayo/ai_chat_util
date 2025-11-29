# 抽象クラス

from abc import ABC, abstractmethod
import base64

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.model import CompletionRequest, CompletionOutput

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMClient(ABC):

    llm_config: LLMConfig = LLMConfig()
    completion_request: CompletionRequest = CompletionRequest()

    @abstractmethod
    async def chat_completion(self, **kwargs) ->  CompletionOutput:
        pass

    @classmethod
    def create_llm_client(cls, llm_config: LLMConfig) -> 'LLMClient':
        if llm_config.llm_provider == "azure_openai":
            return AzureOpenAIClient(llm_config)
        else:
            return OpenAIClient(llm_config)


    def add_image_message_by_path(self, role: str, content:str, image_path: str) -> None:
        """
        Add an image message to the chat history using a local image file path.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_path (str): The local file path to the image.
        """
        if not role or not image_path:
            logger.error("Role and image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.add_image_message(role, content, image_url)

    def add_image_message(self, role: str, content: str, image_url: str) -> None:
        """
        Add an image message to the chat history.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_url (str): The URL of the image to be included in the message.
        """
        
        if not role or not image_url:
            logger.error("Role and image URL must be provided.")
            return
        content_item = [
            {"type": "image_url", "image_url": {"url": image_url}}
            ]
        if content:
            content_item.append({"type": "text", "text": content})

        self.completion_request.messages.append({"role": role, "content": content_item})
        logger.debug(f"Image message added: {role}: {image_url}")

    def append_image_to_last_message_by_path(self, role:str, image_path: str) -> None:
        """
        Append an image to the last message in the chat history using a local image file path.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_path (str): The local file path to the image.
        """
        if not image_path:
            logger.error("Image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.append_image_to_last_message(role, image_url)

    def append_image_to_last_message(self, role:str, image_url: str) -> None:
        """
        Append an image to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_url (str): The URL of the image to append to the last message.
        """
        if not self.completion_request.messages:
            self.completion_request.messages.append({"role": role, "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
            logger.debug("No messages to append to. Added new message.")
            return
        
        last_message = self.completion_request.messages[-1]
        if last_message["role"] != role:
            self.completion_request.messages.append({"role": role, "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
            logger.debug(f"Added new message as last message role '{last_message['role']}'")
            return
        
        # Check if the last content is a list and contains an image item
        if isinstance(last_message["content"], list):
            last_message["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            logger.debug(f"Added new image item to last message: {image_url}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append image.")

    def append_text_to_last_message(self, role:str, additional_text: str) -> None:
        """
        Append additional text to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            additional_text (str): The text to append to the last message.
        """
        if not self.completion_request.messages:
            self.completion_request.messages.append({"role": role, "content": [{"type": "text", "text": additional_text}]})
            logger.debug("No messages to append to. Added new message.")
            return
        last_message = self.completion_request.messages[-1]

        if last_message["role"] != role:
            self.completion_request .messages.append({"role": role, "content": [{"type": "text", "text": additional_text}]})
            logger.debug(f"Added new message as last message role '{last_message['role']}'")
            return

        # Check if the last content is a list and contains a text item
        if isinstance(last_message["content"], list):
            # If no text item found, add a new text item
            last_message["content"].append({"type": "text", "text": additional_text})
            logger.debug(f"Added new text item to last message: {additional_text}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append text.")

    def add_text_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        if not role or not content:
            logger.error("Role and content must be provided.")
            return
        content_item = [{"type": "text", "text": content}]
        self.completion_request.messages.append({"role": role, "content": content_item})
        logger.debug(f"Message added: {role}: {content}")

    def add_user_text_message(self, content: str) -> None:
        """
        Add a user message to the chat history.
        
        Args:
            content (str): The content of the user message.
        """
        self.add_text_message(self.completion_request.user_role_name, content)

    def add_assistant_text_message(self, content: str) -> None:
        """
        Add an assistant message to the chat history.
        
        Args:
            content (str): The content of the assistant message.
        """
        self.add_text_message(self.completion_request.assistant_role_name, content)

    def add_system_text_message(self, content: str) -> None:
        """        Add a system message to the chat history.
        Args:
            content (str): The content of the system message.
        """
        self.add_text_message(self.completion_request.system_role_name, content)

    def get_last_message(self) -> dict:
        """
        Get the last message in the chat history.
        
        Returns:
            Optional[dict]: The last message dictionary or None if no messages exist.
        """
        if self.completion_request.messages:
            last_message = self.completion_request.messages[-1]
            logger.debug(f"Last message retrieved: {last_message}")
            return last_message
        else:
            logger.debug("No messages found.")
            return {}

    def add_messages(self, messages: list[dict]) -> None:
        """
        Add multiple messages to the chat history.
        
        Args:
            messages (list[dict]): A list of message dictionaries to add.
        """
        if not messages:
            logger.error("No messages provided to add.")
            return
        self.completion_request.messages.extend(messages)
        logger.debug(f"Added {len(messages)} messages to chat history.")            

    def to_dict(self) -> dict:
        """
        Convert the chat messages to a dictionary format.
        
        Returns:
            dict: A dictionary representation of the chat messages.
        """
        params = {}
        params["messages"] = self.completion_request.messages
        params["model"] = self.completion_request.model
        if self.completion_request.temperature is not None:
            params["temperature"] = self.completion_request.temperature
        if self.completion_request.response_format is not None:
            params["response_format"] = self.completion_request.response_format
        logger.debug(f"Converting chat messages to dict: {params}")
        return params


class AzureOpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig):
        if llm_config.base_url:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        elif llm_config.api_version and llm_config.endpoint:
            self.client = AsyncAzureOpenAI(api_key=llm_config.api_key, azure_endpoint=llm_config.endpoint, api_version=llm_config.api_version)
        else:
            raise ValueError("Either base_url or both api_version and endpoint must be provided.")

        self.model = llm_config.completion_model

    async def chat_completion(self, **kwargs) -> CompletionOutput:
        
        response = await self.client.chat.completions.create(
            model=self.completion_request.model,
            messages=self.completion_request.messages,
        )
        return CompletionOutput(output=response.choices[0].message.content or "")


class OpenAIClient(LLMClient):
    def __init__(self, llm_config: LLMConfig):
        if llm_config.base_url:
            self.client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        else:
            self.client = AsyncOpenAI(api_key=llm_config.api_key)

        self.model = llm_config.completion_model

    async def chat_completion(self,  **kwargs) -> CompletionOutput:
        response = await self.client.chat.completions.create(
            model=self.completion_request.model,
            messages=self.completion_request.messages,
        )
        return CompletionOutput(output=response.choices[0].message.content or "")
