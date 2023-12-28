from openai import OpenAI
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import stats
import json
from itertools import combinations
import time
import random
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def get_openai_responses(prompt, num_responses=40, max_tokens=100, initial_delay: float = 1,
                         exponential_base: float = 2, jitter: bool = True):
    
    """
    Retrieves a specified number of responses from OpenAI's GPT model.
    
    Args:
        prompt (str): The prompt to send to the model.
        num_responses (int): The number of responses to retrieve.
        max_tokens (int): The maximum number of tokens per response.
        initial_delay (float): Initial delay in seconds before retrying after a rate limit error.
        exponential_base (float): The base for the exponential backoff calculation.
        jitter (bool): Whether to add random jitter to the delay.

    Returns:
        list: A list of responses from the AI model.
    """
    if num_responses > 100:  # Example threshold
        raise ValueError("num_responses is too high. Please reduce the number.")

    delay = initial_delay
    responses = []
    retries = 0
    max_retries = 10

    while len(responses) < num_responses and retries < max_retries:
        try:
            AI_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides recommendations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                n=1,  # Number of completions to generate
            )
            #print(AI_response)
            #need to eventually un comment these
            response_content = AI_response.choices[0].message.content
            responses.append(response_content)
        except Exception as e:
            print(f"Encountered an error: {e}")
            if 'rate limit' in str(e).lower():
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Rate limit exceeded. Waiting for {delay} seconds.")
                time.sleep(delay)
            else:
                print("Encountered a non-rate-limit error. Retrying...")
            retries += 1
            time.sleep(initial_delay)  # Basic delay for non-rate-limit errors

    if retries == max_retries:
        print("Max retries reached. Some responses may not have been retrieved.")

    return responses


def get_bert_embedding(my_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(my_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1)
    
    # Convert to numpy array and check shape
    embedding_np = embedding.numpy()
    #print(f"Embedding shape: {embedding_np.shape}")  # For debugging
    return embedding_np.tolist()


def process_prompt(prompt):
    result_data = []  # Store quant scores for all prompts
    responses = get_openai_responses(prompt)

    prompt_embedding = get_bert_embedding(prompt)  # This returns a list of lists
    response_embeddings = [get_bert_embedding(response) for response in responses]

    cosine_sim_scores = []
    # Create combinations of the prompt embedding and all response embeddings
    response_embeddings = [get_bert_embedding(response) for response in responses]
    cosine_sim_scores = []
    for emb1, emb2 in combinations(response_embeddings, 2):
        # Compute cosine similarity between each pair of response embeddings
        score = cosine_similarity(emb1, emb2)[0][0]
        cosine_sim_scores.append(score)

        # Calculate statistical measures
        mean_score = np.mean(cosine_sim_scores)
        median_score = np.median(cosine_sim_scores)
        mode_result = stats.mode(cosine_sim_scores, keepdims=False)


        # Check if the mode result is a scalar or an array and extract the mode value
        if np.isscalar(mode_result.mode):
            mode_score = mode_result.mode
        else:
            mode_score = mode_result.mode[0] if mode_result.mode.size else None
        average_score = np.average(cosine_sim_scores)
    sample_size = len(cosine_sim_scores)

    quant_scores = {
        "mean": mean_score,
        "median": median_score,
        "mode": mode_score,
        "average": average_score,
        "sample_size": sample_size
    }

    result_data.append({"prompt": prompt, "quant_scores": quant_scores})

    return result_data


if __name__ == "__main__":
    # Example of how to use the module
    prompt_example = "What is the best meal for dinner?"
    a_result = process_prompt(prompt_example)

    with open('results.json', 'w') as f:
        json.dump(a_result, f, indent=2)