def description_prompt_builder_from_response(response):
    return f"""I'm am a person and for the past few days, I've been experiencing {response["disease"]}, 
    that increased on my skin, describe this illness for me ? in 500 characters max"""


def causes_prompt_builder_from_response(response):
    return f"""I'm am a person and for the past few days, I've been experiencing {response["disease"]}, 
    that increased on my skin, what are the causes of it ? in 500 characters max (do not mention the disease definition)"""


def treatement_prompt_builder_from_response(response):
    return f"""I'm am a person and for the past few days, I've been experiencing {response["disease"]}, 
    that increased on my skin, what could be a good treatement for it in 500 characters (do not mention the disease definition) """


def prompt_builder_for_diagnose(symptoms):
    return f"""I'm am a person and for the past few days, I've been experiencing those symptoms {symptoms}, what skin issue or disease can I possibly have, and give me some health recommendations in 500 characters max"""


def parse_llm_response_to_dict(desc, causes, treatement):
    return {
        "description": desc,
        "causes": causes,
        "treatement": treatement,
    }
