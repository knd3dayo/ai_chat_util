import asyncio
from tqdm.asyncio import tqdm_asyncio

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient, OpenAIClient, AzureOpenAIClient
from ai_chat_util.agent.agent_util import MSAIAgentFactory

class LLMBatchClient:
    def __init__(self, config: LLMConfig):
        self.llm_config = config
        if config.llm_provider == "openai":
            self.client = OpenAIClient(config)
        elif config.llm_provider == "azure_openai":
            self.client = AzureOpenAIClient(config)
        else:
            raise ValueError("Unsupported LLM provider")

    async def __process_row(self, row_num: int, prompt: str, input_message: str, agent_mode, progress: tqdm_asyncio) -> tuple[int, str, str]:
        content = f"{prompt}\n{input_message}"
        if agent_mode:
            agent_util = MSAIAgentFactory(llm_config=self.llm_config)

            async with (agent_util.create_agent() as agent):
                response = await agent.run(content)
                progress.update(1)  # Update progress after processing the row
                return (row_num, input_message, response.text)

        messages = [
            {"role": "user", "content": content}
        ]
        response = await self.client._chat_completion(messages=messages)
        progress.update(1)  # Update progress after processing the row
        return (row_num, input_message, response.output)

    async def batch_chat_completion(self, prompt:str, messages: list[str], agent_mode) -> list[str]:

        progress = tqdm_asyncio(total=len(messages), desc="progress")
        # 進捗バーのフォーマット
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


        responses = []

        async with asyncio.Semaphore(5):
            tasks = [self.__process_row(i, prompt, messages, agent_mode, progress) for i, messages in enumerate(messages)]
            responses = await asyncio.gather(*tasks)
            progress.close()
    
        # Sort responses by row number to maintain order
        responses.sort(key=lambda x: x[0])
        return [response for _, _, response in responses]
    
# テストコードは以下の通りです。
if __name__ == "__main__":
    import os
    from ai_chat_util.llm.llm_config import LLMConfig

    async def main():
        llm_config = LLMConfig()
        batch_client = LLMBatchClient(llm_config)
        agent_mode = False

        prompt = "以下の言語のお昼のあいさつを教えて"
        messages = [
            "日本語",
            "英語",
            "中国語"
        ]

        results = await batch_client.batch_chat_completion(prompt, messages, agent_mode)

        for i, res in enumerate(results):
            print(f"Prompt {i+1}: {res}")

    asyncio.run(main())