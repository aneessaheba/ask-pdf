import os

import numpy as np
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from search import search
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"])
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"])
ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

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

test_cases = [
    {"question": "What is the Ego4D dataset?", "ground_truth": "Ego4D is a large scale egocentric video dataset collected from people wearing cameras in their daily lives."},
    {"question": "What tasks does Ego4D benchmark include?", "ground_truth": "Ego4D includes tasks such as episodic memory, forecasting, hand and object interaction, audio visual diarization and social interaction."},
    {"question": "What is the goal of the Ego4D project?", "ground_truth": "The goal is to advance multimodal perception of egocentric video and enable AI to learn from daily life experiences."},
    {"question": "What kind of data does Ego4D contain?", "ground_truth": "Ego4D contains unscripted egocentric video footage captured by people wearing cameras along with annotations and narrations."},
    {"question": "What applications does Ego4D target?", "ground_truth": "Ego4D targets applications in augmented reality, robotics and AI systems that understand human daily activities."}
]

print("Running eval on all questions...\n")
user_inputs, responses, retrieved_contexts, references = [], [], [], []

for tc in test_cases:
    question = tc["question"]
    print(f"Q: {question}")
    chunks = search(question)
    context_texts = [c["text"] for c in chunks]
    answer = get_answer(question, context_texts)
    user_inputs.append(question)
    responses.append(answer)
    retrieved_contexts.append(context_texts)
    references.append(tc["ground_truth"])
    print(f"A: {answer[:100]}...\n")

dataset = Dataset.from_dict({
    "user_input": user_inputs,
    "response": responses,
    "retrieved_contexts": retrieved_contexts,
    "reference": references,
})

print("\nCalculating RAGAS scores...\n")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    # surface real API/parsing errors instead of silently turning them into nan
    raise_exceptions=True,
    # run one call at a time with a generous timeout so a small Gemini quota doesn't trip into 429s
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
