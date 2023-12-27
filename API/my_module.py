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

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def get_openai_responses(prompt, num_responses=40, max_tokens=100, initial_delay: float = 1,
                         exponential_base: float = 2, jitter: bool = True):
    delay = initial_delay
    responses = []
    retries = 0
    max_retries = 10

    while len(responses) < num_responses and retries < max_retries:
        try:
            AI_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides meal recommendations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                n=1,  # Number of completions to generate
            )
            response_content = AI_response["choices"][0]["message"]["content"]
            responses.append(response_content)
        except OpenAI.RateLimitError:
            delay *= exponential_base * (1 + jitter * random.random())  # implementation of exponential backing off
            print(f"Rate limit exceeded. Waiting for {delay} seconds.")
            time.sleep(delay)
            retries += 1

    if retries == max_retries:
        print("Max retries reached. Unable to get a successful response.")

    return responses


def get_bert_embedding(my_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(my_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1)
    return embedding.numpy().tolist()


def process_prompt(prompt):
    result_data = []  # Store quant scores for all prompts
    responses = get_openai_responses(prompt)

    prompt_embedding = get_bert_embedding(prompt)
    response_embeddings = [get_bert_embedding(response) for response in responses]

    cosine_sim_scores = []
    embedding_pairs = list(combinations([prompt_embedding] + response_embeddings, 2))
    for pair in embedding_pairs:
        score = cosine_similarity([pair[0]], [pair[1]])[0][0]
        cosine_sim_scores.append(score)

    mean_score = np.mean(cosine_sim_scores)
    median_score = np.median(cosine_sim_scores)
    mode_score = stats.mode(cosine_sim_scores)
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