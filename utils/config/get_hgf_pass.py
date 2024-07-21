import os 

def getpass():
    return os.environ.get("llm_pass")