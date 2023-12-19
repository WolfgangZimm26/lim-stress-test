import openai
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import stats
import json
from itertools import combinations

# Step 1: Create List of prompts
prompts = ["What is the best meal for dinner?", "What is the best meal for cena?",
           "What is the best meal for abendessen?"]


# Step 2: Collect Responses
def get_openai_responses(prompt, num_responses=40, max_tokens=100):
    responses = []
    while len(responses) < num_responses:
        AI_response = openai.chat.completions.create(
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
    return responses


# Step 3: A Function That Can use Bert Embeddings
def get_bert_embedding(my_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(my_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1)
    return embedding.numpy().tolist()


# Step 4: Collect Embeddings and Calculate Cosine Similarity Scores
all_quant_scores = []  # Store quant scores for all prompts

for prompt in prompts:
    responses = get_openai_responses(prompt)

    prompt_embedding = get_bert_embedding(prompt)
    response_embeddings = [get_bert_embedding(response) for response in responses]

    # Calculate cosine similarity scores for each pair
    cosine_sim_scores = []
    embedding_pairs = list(combinations([prompt_embedding] + response_embeddings, 2))
    for pair in embedding_pairs:
        score = cosine_similarity([pair[0]], [pair[1]])[0][0]
        cosine_sim_scores.append(score)

    # Compute Statistics
    mean_score = np.mean(cosine_sim_scores)
    median_score = np.median(cosine_sim_scores)
    mode_result = stats.mode(cosine_sim_scores)
    mode_score = float(mode_result.mode[0])
    average_score = np.average(cosine_sim_scores)
    sample_size = len(cosine_sim_scores)

    quant_scores = {
        "mean": mean_score,
        "median": median_score,
        "mode": mode_score,
        "average": average_score,
        "sample_size": sample_size
    }

    # Save quant scores for this prompt
    all_quant_scores.append({"prompt": prompt, "quant_scores": quant_scores})

# Step 5: Save Results
result_data = {
    "all_quant_scores": all_quant_scores
}

with open('results.json', 'w') as f:
    json.dump(result_data, f, indent=2)
