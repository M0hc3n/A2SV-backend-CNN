from flask import jsonify, Flask, request, make_response

from utils.llama_handler import ask_llm_handler, get_tip_from_llm

from errors.non_formatted_input import input_non_valid_error
from errors.status import status
import requests

import os
import pickle

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

from utils.llm_utils import (
    description_prompt_builder_from_response,
    causes_prompt_builder_from_response,
    treatement_prompt_builder_from_response,
)

app = Flask(__name__)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

def get_urls():
    return [
    "https://my.clevelandclinic.org/health/diseases/12233-acne",
    "https://my.clevelandclinic.org/health/diseases/14148-actinic-keratosis",
    "https://my.clevelandclinic.org/health/diseases/24299-atopic-dermatitis",
    "https://my.clevelandclinic.org/health/diseases/15855-bullous-pemphigoid",
    "https://my.clevelandclinic.org/health/diseases/15071-cellulitis", 
    "https://my.clevelandclinic.org/health/diseases/9998-eczema"
    "https://my.clevelandclinic.org/health/diseases/22510-viral-exanthem-rash", 
    "https://my.clevelandclinic.org/health/diseases/21753-hair-loss", 
    "https://my.clevelandclinic.org/health/diseases/22855-herpes-simplex", 
    "https://my.clevelandclinic.org/health/symptoms/11014-skin-discoloration", 
    "https://my.clevelandclinic.org/health/symptoms/23163-lupus-rash", 
    "https://my.clevelandclinic.org/health/diseases/14391-melanoma", 
    "https://my.clevelandclinic.org/health/diseases/11303-toenail-fungus", 
    "https://my.clevelandclinic.org/health/diseases/6866-psoriasis", 
    "https://my.clevelandclinic.org/health/diseases/4567-scabies", 
    "https://my.clevelandclinic.org/health/diseases/21721-seborrheic-keratosis", 
    "https://my.clevelandclinic.org/health/diseases/24386-systemic-mastocytosis", 
    "https://my.clevelandclinic.org/health/diseases/4560-ringworm", 
    "https://my.clevelandclinic.org/health/diseases/24063-tinea-manuum", 
    "https://my.clevelandclinic.org/health/symptoms/22454-hives-in-children", 
    "https://my.clevelandclinic.org/health/diseases/23357-stasis-ulcer", 
    "https://my.clevelandclinic.org/health/diseases/12101-vasculitis", 
    "https://my.clevelandclinic.org/health/diseases/15045-warts"
    ]


def load_document_from_urls(urls):
    loader = WebBaseLoader(urls)
    documents = loader.load()
    
    return documents



@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["file"]
    except KeyError:
        return make_response(jsonify(input_non_valid_error), status["input_not_valid"])

    try:
        # Forward the file to the cv:5001/detect endpoint
        files = {"file": (file.filename, file.stream, file.content_type)}
        cv_response = requests.post("http://localhost:5001/detect", files=files)
        cv_response.raise_for_status()
        response = cv_response.json()
    except requests.RequestException as e:
        return make_response(
            jsonify(
                {"error": "Failed to communicate with CV service", "details": str(e)}
            ),
            500,
        )

    # output_dir = "./utils/models"

    documents = load_document_from_urls(get_urls())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(split_docs, embeddings)

    hf_endpoint = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    #     token=huggingface_api_key,
        model_kwargs={"temperature": 0.7}
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a knowledgeable dermatologist providing expert advice based on the provided context. 

        Context: {context}

        Question: {question}

        Please provide a detailed and accurate answer based on the context. Include any relevant information, potential diagnoses, and recommended treatments or next steps. Your response should be professional, clear, and concise, mimicking how a dermatologist would address a patient's concerns.

        Answer:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=hf_endpoint,
        retriever=vector_store.as_retriever(),
        prompt_template=prompt_template
    )



    # faiss_save_path = os.path.join(output_dir, "faiss_index")
    # vector_store = FAISS.load_local(
    #     faiss_save_path,
    #     HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    #     allow_dangerous_deserialization=True
    # )  # Replace with your embeddings

    # chain_save_path = os.path.join(output_dir, "qa_chain.pkl")
    # with open(chain_save_path, "rb") as f:
    #     qa_chain = pickle.load(f)

    description = qa_chain.run(description_prompt_builder_from_response(response))
    causes = qa_chain.run(causes_prompt_builder_from_response(response))
    treatement = qa_chain.run(treatement_prompt_builder_from_response(response))

    description = description.strip()
    causes = causes.strip()
    treatement = treatement.strip()

    # llm_response = ask_llm_handler(False, response=response)

    return make_response(jsonify({"description":description,"causes":causes,"treatement":treatement}))

    # return make_response(
    #     jsonify({"data": {**response, "llm_response": llm_response}}), status["success"]
    # )


@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.json
        symptoms = data.get("symptoms")

    except KeyError:
        return make_response(jsonify(input_non_valid_error), status["input_not_valid"])

    response = ask_llm_handler(True, symptoms=symptoms)

    return make_response(
        jsonify({"data": {"llm_response": response}}), status["success"]
    )


@app.route("/tip", methods=["GET"])
def get_tip():
    response = get_tip_from_llm()

    return make_response(
        jsonify({"data": {"llm_response": response}}), status["success"]
    )


if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
