from chatsky_llm_autoconfig.algorithms.base import TopicGraphGenerator
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.schemas import DialogueGraph
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from langchain_core.language_models.chat_models import BaseChatModel


@AlgorithmRegistry.register(input_type=str, output_type=BaseGraph)
class CycleGraphGenerator(TopicGraphGenerator):
    """Generator specifically for topic-based cyclic graphs"""

    def __init__(self):
        super().__init__()

    def invoke(self, model: BaseChatModel, prompt: PromptTemplate, **kwargs) -> BaseGraph:
        """
        Generate a cyclic dialogue graph based on the topic input.

        Args:
            model (BaseChatModel): The model to use for generation
            prompt (PromptTemplate): Prepared prompt template
            **kwargs: Additional arguments for formatting the prompt

        Returns:
            BaseGraph: Generated Graph object with cyclic structure
        """
        # Создаем цепочку: промпт -> модель -> парсер
        parser = JsonOutputParser(pydantic_object=DialogueGraph)
        chain = prompt | model | parser

        # Передаем kwargs как входные данные для цепочки
        return Graph(chain.invoke(kwargs))

    async def ainvoke(self, *args, **kwargs):
        """
        Async version of invoke - to be implemented
        """
        pass
