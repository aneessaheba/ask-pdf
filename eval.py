import os

import numpy as np
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from datasets import Dataset
from search import search
from groq import Groq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# use local llama3.1 as judge — free, no quota, no API key
ollama_llm = ChatOllama(model="llama3.1")
ollama_embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

ragas_llm = LangchainLLMWrapper(ollama_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

# use groq for answer generation
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question, contexts):
    context_text = ""
    for i, c in enumerate(contexts):
        context_text += f"[{i+1}] {c}\n\n"
    prompt = f"Answer the question using only the context below.\nContext:\n{context_text}\nQuestion: {question}\nAnswer:"
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# test questions and ground truth answers
test_cases = [
    {"question": "What is the Ego4D dataset?", "ground_truth": "Ego4D is a large scale egocentric video dataset collected from people wearing cameras in their daily lives."},
    {"question": "What tasks does Ego4D benchmark include?", "ground_truth": "Ego4D includes tasks such as episodic memory, forecasting, hand and object interaction, audio visual diarization and social interaction."},
    {"question": "What is the goal of the Ego4D project?", "ground_truth": "The goal is to advance multimodal perception of egocentric video and enable AI to learn from daily life experiences."},
    {"question": "What kind of data does Ego4D contain?", "ground_truth": "Ego4D contains unscripted egocentric video footage captured by people wearing cameras along with annotations and narrations."},
    {"question": "What applications does Ego4D target?", "ground_truth": "Ego4D targets applications in augmented reality, robotics and AI systems that understand human daily activities."}
]

print("Running eval on all questions...\n")
questions, answers, contexts, ground_truths = [], [], [], []

for tc in test_cases:
    question = tc["question"]
    print(f"Q: {question}")
    chunks = search(question)
    context_texts = [c["text"] for c in chunks]
    answer = get_answer(question, context_texts)
    questions.append(question)
    answers.append(answer)
    contexts.append(context_texts)
    ground_truths.append(tc["ground_truth"])
    print(f"A: {answer[:100]}...\n")

dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

print("\nCalculating RAGAS scores using local Llama3.1...\n")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False,
    run_config=RunConfig(max_workers=1, timeout=600),
)

print("\n=============================")
print("      RAGAS EVAL RESULTS     ")
print("=============================")
print(f"Faithfulness:      {np.nanmean(result['faithfulness']):.2f}")
print(f"Answer Relevancy:  {np.nanmean(result['answer_relevancy']):.2f}")
print(f"Context Precision: {np.nanmean(result['context_precision']):.2f}")
print(f"Context Recall:    {np.nanmean(result['context_recall']):.2f}")
print("=============================")
print("Score guide: 0.0 = bad  |  0.5 = okay  |  1.0 = perfect")