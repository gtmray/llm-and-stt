import os
from typing import Dict, Any, Tuple
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())


class AzureOpenAIModel:
    def __init__(
        self, temperature: float = 0.2, verbose: bool = False, **kwargs
    ):
        """
        Initialize AzureOpenAIModel with customizable parameters for the LLM.

        Args:
            temperature (float): Temperature setting for model responses.
            verbose (bool): Verbosity flag for logging output.
            **kwargs: Additional arguments for configuring the AzureChatOpenAI instance.
        """
        self.temperature = temperature
        self.verbose = verbose
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=temperature,
            verbose=verbose,
            **kwargs,
        )

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the model instance.
        """
        return (
            f"AzureOpenAIModel(temperature={self.temperature}, verbose={self.verbose}, "
            f"azure_deployment={os.getenv('AZURE_OPENAI_DEPLOYMENT')}, "
            f"azure_endpoint={os.getenv('AZURE_OPENAI_ENDPOINT')})"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the model instance.
        """
        return f"AzureOpenAIModel with temperature={self.temperature} and verbose={self.verbose}"

    @staticmethod
    def _create_prompt(
        system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """
        Private static method to create prompt template from system and human messages.
        """
        messages = [("system", system_prompt), ("human", human_prompt)]
        return ChatPromptTemplate.from_messages(messages)

    def run_model(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        input_message: Dict[str, Any] = {},
    ) -> str:
        """
        Run a model based on a simple prompt-response with system and human inputs.
        """
        prompt = self._create_prompt(system_prompt, human_prompt)
        try:
            chain = prompt | self.llm
            result = chain.invoke(input_message)
            return result.content
        except Exception as e:
            print(f"Error in running model: {e}")
            return ""

    def run_rag_model(
        self,
        retriever: Any,
        question: str = "",
        system_prompt: str = "",
        human_prompt: str = "",
    ) -> Tuple[str, str]:
        """
        Run a retrieval-augmented generation (RAG) model, combining retrieved information with prompts.
        """
        prompt = self._create_prompt(system_prompt, human_prompt)
        format_docs = lambda docs: "\n\n".join(
            doc.page_content for doc in docs
        )

        try:
            input_message = {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            rag_chain = input_message | prompt | self.llm | StrOutputParser()
            response = rag_chain.invoke(question)
            return response
        except Exception as e:
            print(f"Error in running RAG model: {e}")
            return ""
