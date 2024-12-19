# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:10:13 2024

@author: hyeongyu
"""
import json
import os
import cmlapi
import sys
import gradio as gr
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional
import boto3
from botocore.config import Config
from huggingface_hub import hf_hub_download

import chromadb
from chromadb.utils import embedding_functions


AWS_FOUNDATION_MODEL_ID = os.environ.get("AWS_FOUNDATION_MODEL_ID")

EMBEDDING_MODEL_REPO = os.environ.get("HF_EMBEDDING_MODEL_REPO")
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_REPO.split('/')[-1]
CHROMA_COLLECTION_NAME = "cml-default"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_REPO)

# ChromaDB 연결
chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

try:
    chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function)
    print("ChromaDB collection loaded.")
except:
    chroma_collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function)
    print("ChromaDB collection created.")

# ChromaDB에서 가장 유사한 문서 검색
def get_chroma_context(query, top_k=3):
    print("Querying ChromaDB for similar documents...")
    results = chroma_collection.query(query_texts=[query], n_results=top_k)
    
    # 검색된 문서들을 하나의 문자열로 결합
    if results["documents"]:
        context = "\n\n".join([" ".join(doc) for doc in results["documents"]])
        print(f"Retrieved context: {context}")
        return context
    else:
        print("No relevant documents found.")
        return ""


if os.environ.get("LOCAL_MODEL_ACCESS_KEY") == "":
    client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
    projects = client.list_projects(include_public_projects=True, search_filter=json.dumps({"name": "LLM Model deploy for Hands on Lab"}))
    project = projects.projects[0]
    
    ## Here we assume that only one model has been deployed in the project, if this is not true this should be adjusted (this is reflected by the placeholder 0 in the array)
    model = client.list_models(project_id=project.id)
    selected_model = model.models[0]
    
    ## Save the access key for the model to the environment variable of this project
    MODEL_ACCESS_KEY = selected_model.access_key
else:
    MODEL_ACCESS_KEY = os.environ.get("LOCAL_MODEL_ACCESS_KEY")

MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

# Double check default region
if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    
## Setup Bedrock client:
def get_bedrock_client(
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)


    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(
      region=os.environ.get("AWS_DEFAULT_REGION", None))


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses, 
        title="Enterprise Custom Knowledge Base Chatbot",
        description = DESC,
        additional_inputs=[gr.Radio(['Local Mistral 7B', 'AWS Bedrock Mistral 7B'], label="Select Foundational Model", value="AWS Bedrock Mistral 7B", visible=True), 
                           gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Slider(minimum=0.01, maximum=5.0, step=0.01, value=1.2, label="Select Repetition penalty (Penalty for repetition)"),
                           gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"),
                           gr.Radio(['None', 'ChromaDB'], label="Vector Database Choices", value="None")],
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

    
# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, repetition_penalty, token_count, vector_db):
    
    # Process chat history
    #chat_history_string = '; '.join([strng for xchng in history for strng in xchng])
    #print(f"Chat so far {history}")
    
    context_chunk = get_chroma_context(message) if vector_db == "ChromaDB" else ""

    if 'mistral' in model.lower():
        if vector_db == "None":
            context_chunk = ""
        response = get_mistral_response_with_context(model, message, context_chunk, temperature, repetition_penalty, token_count)
     
    elif 'claude' in model.lower():
        if vector_db == "None":
            # No context call Bedrock
            context_chunk = ""
        response = get_bedrock_claude_response_with_context(message, context_chunk, temperature, repetition_penalty, token_count)
    
    # Stream output to UI
    for i in range(len(response)):
        time.sleep(0.01)
        yield response[:i+1]

  
  
def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    

# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()


def get_bedrock_claude_response_with_context(question, context, temperature, token_count):
    
    # Supply different instructions, depending on whether or not context is provided
    if context == "":
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    
    <question>{{QUESTION}}</question>
                    Assistant:"""
    else:
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please read the text provided between the tags <text></text> and provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    <text>{{CONTEXT}}</text>
    
    <question>{{QUESTION}}</question>
                    Assistant:"""

    
    # Replace instruction placeholder to build a complete prompt
    full_prompt = instruction_text.replace("{{QUESTION}}", question).replace("{{CONTEXT}}", context)
    
    # Model expects a JSON object with a defined schema
    body = json.dumps({"prompt": full_prompt,
             "max_tokens_to_sample":int(token_count),
             "temperature":float(temperature),
             "top_k":250,
             "top_p":1.0,
             "stop_sequences":[]
              })

    # Provide a model ID and call the model with the JSON payload
    modelId = 'anthropic.claude-v2:1'
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    print("Model results successfully retreived")
    
    result = response_body.get('completion')
    #print(response_body)
    
    return result

    
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_mistral_response_with_context(model, question, context, temperature, repetition_penalty, token_count):
    # Prompt 구조 통일
    if context:
        prompt = f"""
        ### Instruction:
        You are a helpful and honest assistant. If you are unsure about an answer, truthfully say "I don't know".
        
        ### Context:
        {context}\n
        
        ### Question:
        {question}\n
        
        ### Response:
        """
    else:
        prompt = f"""
        ### Instruction:
        You are a helpful and honest assistant. If you are unsure about an answer, truthfully say "I don't know".
        
        ### Question:
        {question}\n
        
        ### Response:
        """
    if 'aws' in model.lower():
        try:
            data =  {"prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": token_count}
            response = boto3_bedrock.invoke_model(body=json.dumps(data), modelId=AWS_FOUNDATION_MODEL_ID, accept='application/json', contentType='application/json')
            response_body = json.loads(response.get('body').read())
            print("Model results successfully retreived")
            result = response_body.get('outputs')
            print(f"Request: {data}")
            print(f"Response: {result}")
            return result[0]['text']
        except Exception as e:
            print(e)
            return str(e)
    else: 
        try:
            # 요청 생성 및 반환
            data = {
                "request": {"prompt": prompt,
                            "temperature": temperature,
                            "max_new_tokens": token_count,
                            "repetition_penalty": repetition_penalty}
            }
            response = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
            result = response.json()['response']['result'].split("### Response:")[-1].strip()  
            
            print(f"Request: {data}")
            print(f"Response: {result}")
            return result
    
        except Exception as e:
            print(e)
            return str(e)

if __name__ == "__main__":
    main()
