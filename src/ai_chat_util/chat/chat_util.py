import json
from typing import ClassVar
from pydantic import BaseModel, Field
import copy
import tiktoken

from ai_chat_mcp.llm.llm_util import LLMClient, OpenAIProps, CompletionRequest, CompletionResponse, ChatMessageItem

import ai_chat_mcp.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class ChatRequestContext(BaseModel):

    # split_mode
    split_mode_name_none: ClassVar[str] = "None"
    split_mode_name_normal: ClassVar[str] = "NormalSplit"
    split_mode_name_split_and_summarize: ClassVar[str] = "SplitAndSummarize"

    # メッセージを分割するモード。分割しない場合は"None"、通常分割は"NormalSplit"、分割して要約は"SplitAndSummarize"
    split_mode: str = Field(default="None", description="Mode to split messages. 'None' for no split, 'NormalSplit' for normal split, 'SplitAndSummarize' for split and summarize.")
    # メッセージ分割時のトークン数の上限
    split_token_count: int = Field(default=2000, description="Maximum token count for message splitting.")
    # 複数画像URLがある場合に、1つのリクエストに含める最大画像数。分割しない場合は0を設定
    max_images_per_request: int = Field(default=0, description="Maximum number of images to include in a single request. Set to 0 if not splitting.")
    # SplitAndSummarizeモード時の要約用プロンプトテキスト
    summarize_prompt_text: str = Field(default="Summarize the content concisely.", description="Prompt text for summarization in SplitAndSummarize mode.")
    # プロンプトテンプレートテキスト. 分割モードがNone以外の場合に使用. 分割した各メッセージの前に付与する。
    # 分割モードがNone以外の場合は、各パートはこのプロンプトの指示に従うため、必ず設定すること。
    prompt_template_text: str = Field(default="", description="Prompt template text. Used when split mode is not 'None'. This text is prepended to each split message. When split mode is not 'None', this must be set to guide each part according to the prompt's instructions.")

class ChatUtil:

    @classmethod
    def split_message(cls, original_message: list[str], model: str, split_token_count: int) -> list[str]:
        # token_countがsplit_token_countを超える場合は分割する
        result_message_list = []
        current_message = ""
        for line in original_message:
            line_token_count = cls.get_token_count(model, line)
            current_message_token_count = cls.get_token_count(model, current_message)
            if current_message_token_count + line_token_count > split_token_count:
                # current_messageをresult_message_listに追加する
                result_message_list.append(current_message)
                # current_messageを初期化する
                current_message = line + "\n"
            else:
                current_message += line + "\n"
        # 最後のcurrent_messageをresult_message_listに追加する
        if len(current_message) > 0:
            result_message_list.append(current_message)

        return result_message_list

    @classmethod
    def __get_last_message_image_urls(cls, message_dict: ChatMessageItem) -> tuple[int, list[str]]:
        '''
        message_dictのmessagesの最後のimage_url要素を取得する
    
        '''
        image_urls = []
        message_index = -1
        # "messages"のimage_url要素を取得する
        for i in range(0, len(message_dict.content)):
            content = message_dict.content[i]
            if content.type == "image_url":
                if content.image_url:
                    image_urls.append(content.image_url["url"])
                message_index = i

        return message_index, image_urls

    @classmethod
    def __get_last_message_text(cls, message_dict: ChatMessageItem) -> tuple[int, str]:
        '''
        message_dictのmessagesの最後のtext要素を取得する
    
        '''
        message_index = -1
        # "messages"のtext要素を取得する       
        for i in range(0, len(message_dict.content)):
            if message_dict.content[i].type == "text":
                message_index = i
                break
        # message_indexが-1の場合はエラーをraiseする
        if message_index == -1:
            raise ValueError("last_text_content_index is -1")
        # queryとして最後のtextを取得する
        last_message = message_dict.content[message_index].text
        return message_index, last_message if last_message else ""

    @classmethod
    async def __pre_process_input(
            cls, client: LLMClient, request_context: ChatRequestContext, original_chat_request: CompletionRequest
            ) -> tuple[list[CompletionRequest], list[dict]]:
        '''
        メッセージ分割、ベクトル検索を実行する
        split_modeがNoneの場合はメッセージ分割を実行しない。
        split_modeがNone以外の場合はメッセージ分割を実行する。また、image_urlも分割する
        その後、rag_modeがNone以外の場合はベクトル検索を実行する
        返り値は、分割後のChatRequestのリストとベクトル検索結果のリスト
        '''


        # 結果格納用のChatRequestのリストを作成する
        result_chat_request_list: list[CompletionRequest] = []

        # pre_process_inputを実行する
        chat_request = copy.deepcopy(original_chat_request)
        # chat_requestのmessagesの最後の要素を取得する
        last_message_dict = chat_request.messages.pop()
        if not last_message_dict:
            raise ValueError("No last message found in input_dict")


        # "messages"の最後のtext要素を取得する       
        last_text_content_index, original_last_message = cls.__get_last_message_text(last_message_dict)

        # "messages"の最後のimage_url要素を取得する       
        last_images_content_index, image_urls = cls.__get_last_message_image_urls(last_message_dict)

        # request_contextのSplitModeがNone以外の場合はoriginal_last_messageを改行毎にtokenをカウントして、
        # split_token_countを超える場合は分割する

        result_documents_dict = {}  # Ensure this is always defined

        # split_modeがNoneの場合は、context_message, original_last_message, vector_search_result_messageを結合して
        # chat_requestのmessagesに追加する
        if request_context.split_mode == ChatRequestContext.split_mode_name_none:
            chat_request.add_text_message(CompletionRequest.user_role_name, 
                f"{request_context.prompt_template_text}\n{original_last_message}\n\n")
            # image_urlが存在する場合はchat_requestに追加する
            for image_url in image_urls:
                chat_request.append_image_to_last_message(CompletionRequest.user_role_name, image_url)

            # result_chat_request_listにchat_requestを追加する
            result_chat_request_list.append(chat_request)
            return result_chat_request_list, [ value for value in result_documents_dict.values()]

        # SplitModeがNone以外の場合はoriginal_last_messageを分割する
        splited_messages = cls.split_message(original_last_message.split("\n"), client.props.completion_model, request_context.split_token_count)
        for i in range(0, len(splited_messages)):
            # 分割したメッセージを取得する毎に、プロンプトテンプレートと関連情報を取得する
            target_message = splited_messages[i]
            # chat_requestをdeepcopyする
            result_chat_request = copy.deepcopy(chat_request)
            
            # result_chat_requestのmessagesにtext_messageを追加する
            result_chat_request.add_text_message(CompletionRequest.user_role_name, 
                f"{request_context.prompt_template_text}\n{target_message}\n\n")
            # result_chat_request_listにresult_chat_requestを追加する
            result_chat_request_list.append(result_chat_request)

        # SplitModeがNone以外の場合はimage_urlsを分割する。
        splited_image_urls = []
        if request_context.split_mode != ChatRequestContext.split_mode_name_none:
            max_images_per_request = request_context.max_images_per_request
            if len(image_urls) > 0 and max_images_per_request > 0:
                for i in range(0, len(image_urls), max_images_per_request):
                    splited_image_urls.append(image_urls[i:i + max_images_per_request])
        else:
            splited_image_urls = [image_urls]
        
        # splited_image_urls毎にchat_requestを作成してresult_chat_request_listに追加する
        for i in range(0, len(splited_image_urls)):
            image_urls = splited_image_urls[i]
            if len(image_urls) == 0:
                continue
            # chat_requestをdeepcopyする
            result_chat_request = copy.deepcopy(chat_request)
            # image_urlsをresult_chat_requestに追加する
            for image_url in image_urls:
                message = f"{request_context.prompt_template_text}\n\n"
                result_chat_request.add_image_message(CompletionRequest.user_role_name, message, image_url)
            # result_chat_request_listにresult_chat_requestを追加する
            result_chat_request_list.append(result_chat_request)
        return result_chat_request_list, [ value for value in result_documents_dict.values()]

    @classmethod
    async def __post_process_output_async(cls, client: LLMClient, request_context: ChatRequestContext, 
                            input_dict: CompletionRequest, chat_output_list: list[CompletionResponse],
                            ) -> CompletionResponse:

        # RequestContextのSplitModeがNormalSplitの場合はchat_result_dict_listのoutputを結合した文字列とtotal_tokensを集計した結果を返す
        if request_context.split_mode == ChatRequestContext.split_mode_name_normal:
            output_list = []
            total_tokens = 0

            for i in range(0, len(chat_output_list)):
                logger.debug(f"chat_output_list[{i}]: {chat_output_list[i]}")
                output_dict = {}
                output_dict["part"] = i + 1
                output_dict["output"] = chat_output_list[i].output
                output_list.append(output_dict)
                total_tokens += chat_output_list[i].total_tokens
            
            output_text = json.dumps(output_list, ensure_ascii=False, indent=2)
            return CompletionResponse(output=output_text, total_tokens=total_tokens)

        # RequestContextのSplitModeがSplitAndSummarizeの場合はSummarize用のoutputを作成する
        if request_context.split_mode == ChatRequestContext.split_mode_name_split_and_summarize:
            summary_prompt_text = ""
            if len(request_context.prompt_template_text) > 0:
                summary_prompt_text = f"""
                The following text is a document that was split into several parts, and based on the instructions of [{request_context.prompt_template_text}], 
                the AI-generated responses were combined. 
                {request_context.prompt_template_text}
                """
            else:
                summary_prompt_text = """
                The following text is a document that has been divided into several parts, with AI-generated responses combined.
                {request_context.PromptTemplateText}
                """
            
            summary_input =  summary_prompt_text + "\n".join([chat_output.output for chat_output in chat_output_list])
            total_tokens = sum([chat_output.total_tokens for chat_output in chat_output_list])
            # openai_chatの入力用のdictを作成する
            summary_chat_request = CompletionRequest.create_simple_request(client.props.completion_model, summary_input, input_dict.temperature,  False)
            # chatを実行する
            summary_chat_output = await client.run_completion_async(summary_chat_request)
            # total_tokensを更新する
            summary_chat_output.total_tokens = total_tokens + summary_chat_output.total_tokens
            return summary_chat_output
        else:
            # RequestContextのSplitModeがNoneの場合はoutput_dictの1つ目の要素を返す
            chat_output = chat_output_list[0]
            return chat_output

    @classmethod
    async def run_openai_chat_async(
        cls, input_dict: CompletionRequest, request_context: ChatRequestContext
    ) -> CompletionResponse:
    
        props = OpenAIProps()
        # modelを取得する
        model = input_dict.model
        if model:
            props.completion_model = model
        
        # 最後のメッセージの分割処理、ベクトル検索処理を行う
        # OpenAIClientを取得する
        client = LLMClient(props)

        pre_processed_chat_request_list, docs_list = await cls.__pre_process_input(
            client, request_context, input_dict
            )
        chat_result_dict_list = []

        for pre_processed_chat_request in  pre_processed_chat_request_list:

            chat_result_dict = await client.run_completion_async(pre_processed_chat_request)
            # chat_result_dictをchat_result_dict_listに追加する
            chat_result_dict_list.append(chat_result_dict)

            # 0.5秒待機する
            import time
            time.sleep(0.5)

        # post_process_outputを実行する
        result_dict = await cls.__post_process_output_async(
            client, request_context, input_dict, chat_result_dict_list
        )
        return result_dict
    

    @classmethod
    def get_token_count(cls, model: str, input_text: str) -> int:
        # completion_modelに対応するencoderを取得する
        # 暫定処理 
        # "gpt-4.1-": "o200k_base",  # e.g., gpt-4.1-nano, gpt-4.1-mini
        # "gpt-4.5-": "o200k_base", # e.g., gpt-4.5-preview
        if model.startswith("gpt-41") or model.startswith("gpt-4.1") or model.startswith("gpt-4.5"):
            encoder = tiktoken.get_encoding("o200k_base")
        else:
            encoder = tiktoken.encoding_for_model(model)
        # token数を取得する
        return len(encoder.encode(input_text))
   

