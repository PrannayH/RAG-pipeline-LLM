from langchain.llms import GPT4All, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.schema import Document
import json

LLM_MAPPING = {
    "GPT4All": (
        GPT4All,
        {
            "model": "models/ggml-gpt4all-j-v1.3-groovy.bin",
            "max_tokens": 1,
            "n_batch": 1,
            "verbose": False,
            "backend": "gptj",
        },
    )
}


LLM_PROMPT_MAPPING = {}


def json_to_langchain_document(chunk_list: list):
    docs = list()
    for chunk in chunk_list:
        docs.append(Document(page_content=chunk, metadata={}))

    return docs


def model_selector(config: dict):
    """Will return the model on the basis of given config"""
    pass


class llm:
    def __init__(
        self, model_name: str = "GPT4All", task: str = "qa", verbose: bool = True
    ):
        self.llm_class, self.llm_arg = LLM_MAPPING[model_name]
        self.llm = self.llm_class(**self.llm_arg)
        self.verboseprint = print if verbose else lambda *a: None
        self.prompt = PromptTemplate.from_template(
            "Based on the given context, answer the given Question. \n context: {context} \n Question: {query} \n Answer: "
        )

    def _get_context(self, docs: list[Document]) -> str:
        context = ""
        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    def query(self, query: str, chunk_list: list) -> str:
        """returns the LLM response"""

        docs = json_to_langchain_document(chunk_list)
        context = self._get_context(docs)
        prompt = self.prompt.format(query=query, context=context)
        print(prompt)
        print(f" prompt length = {len(prompt)}")
        response = self.llm(prompt)

        self.verboseprint(f"LLM: Response: {response} \n")
        return response
