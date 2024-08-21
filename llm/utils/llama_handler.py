from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from utils.config.get_hgf_pass import getpass

from utils.llm_utils import (
    description_prompt_builder_from_response,
    causes_prompt_builder_from_response,
    treatement_prompt_builder_from_response,
    parse_llm_response_to_dict,
    prompt_builder_for_diagnose,
    prompt_builder_for_tip,
    prompt_builder_for_tip_detail,
    parse_llm_tip,
)

from utils.config.llm_config import llm_config

HUGGINGFACEHUB_API_TOKEN = "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"


def ask_llm(question):
    prompt = PromptTemplate.from_template(llm_config["template"])

    llm = HuggingFaceEndpoint(
        repo_id=llm_config["repo_id"],
        max_length=llm_config["max_length"],
        temperature=llm_config["temperature"],
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    llm_chain = prompt | llm

    return llm_chain.invoke({"question": question})


def ask_llm_handler(diagnose, response="", symptoms=""):
    if diagnose:
        response = ask_llm(question=prompt_builder_for_diagnose(symptoms))

        return response
    else:
        description = ask_llm(description_prompt_builder_from_response(response))
        causes = ask_llm(causes_prompt_builder_from_response(response))
        treatement = ask_llm(treatement_prompt_builder_from_response(response))

        return parse_llm_response_to_dict(
            desc=description, causes=causes, treatement=treatement
        )


def get_tip_from_llm():
    tip = ask_llm(prompt_builder_for_tip())
    tip_detailed = ask_llm(prompt_builder_for_tip_detail(tip))

    return parse_llm_tip(tip=tip, detailed=tip_detailed)
