# Adaptive Chatbot for UPSC Aspirants - Project Plan

This document outlines the approach and step-by-step process for building the adaptive chatbot as per the assignment requirements.

## Phase 1: Problem Deconstruction and Data Strategy

This phase focuses on deeply understanding the core problems and creating a high-quality dataset, which is crucial for training a reliable model.

### 1.1. Deconstructing the Core Tasks

The problem can be broken down into two primary, interconnected tasks:

*   **Contextual Query Expansion:** This is a **context resolution** task. The goal is to transform a short, context-dependent user query (e.g., "what about uk") into a complete, standalone question (e.g., "who is pm of uk and what are his duties?"). This involves resolving:
    *   **Anaphora:** Replacing pronouns like "his," "her," "it," "they" with the actual entities from the conversation history (e.g., "his duties" -> "duties of narendra modi").
    *   **Ellipsis:** Filling in missing information based on the conversational context (e.g., "what about uk" implies the previous questions about the PM and their duties should be applied to the new entity, "UK").

*   **Hierarchical Topic Tagging:** This is a **text classification** task. The goal is to assign a predefined, 2-level topic to the user's query (e.g., `Politics - India`). The `GENERAL` topic will serve as a catch-all for conversational filler and chitchat that doesn't pertain to UPSC subjects.

### 1.2. The Role of Models: Why a Hybrid Approach is Best

As hinted in the problem description, relying solely on a large language model (LLM) via API calls is not the optimal strategy. A more scientific and robust approach involves training smaller, specialized models.

*   **LLMs (for Data Generation):** We can leverage powerful LLMs like GPT-4 or Claude for what they excel at: generating vast amounts of creative and diverse synthetic data. This is a perfect use case for bootstrapping our dataset.
*   **Trained Models (for Production):** For the actual expansion and tagging tasks, we will fine-tune a smaller, pre-trained transformer model (like T5 for expansion and BERT for classification). This provides:
    *   **Reliability & Consistency:** The model's behavior will be deterministic and tailored to our specific topics and expansion formats.
    *   **Speed:** Smaller models have significantly lower latency, which is critical for a real-time user experience.
    *   **Cost-Effectiveness:** It avoids per-query API costs, making the solution scalable.
    *   **Control:** We have full control over the model's capabilities and can prevent it from generating off-topic or incorrect content.

### 1.3. Data Creation: The Scientific Approach

Since no dataset is provided, we must create one from scratch. This will be done systematically.

**Step 1: Define the Data Schema**
We will use a structured JSONL format, where each line is a JSON object representing a single training example. This schema will capture all necessary information.

```json
{
  "conversation_id": "string",
  "turn_id": "integer",
  "history": [
    {"role": "user", "content": "user's previous message"},
    {"role": "bot", "content": "bot's previous response"}
  ],
  "current_query": "The user's latest, context-dependent query",
  "expanded_query": "The ground-truth, expanded, standalone query",
  "topic": "The ground-truth 2-level topic label"
}
```

**Step 2: Data Sourcing and Generation**
We will create a diverse dataset using a multi-pronged approach:

1.  **Manual "Gold" Set (Seed Data):** Manually author ~100 high-quality, diverse conversation snippets. These will cover various UPSC topics (Polity, History, Geography, Economy), different types of contextual dependencies, and topic shifts. This set will serve as our quality benchmark.
2.  **LLM-Powered Synthetic Generation:** Use the manually created set as few-shot examples to prompt a large LLM. We will ask it to generate thousands of similar conversational exchanges. This will rapidly scale our dataset.
3.  **Data Augmentation:** To improve model robustness, we will programmatically augment the generated data by:
    *   **Paraphrasing:** Rephrasing user queries.
    *   **Entity Substitution:** Swapping entities (e.g., "India" -> "USA", "PM" -> "President").
    *   **Noise Injection:** Adding typos or grammatical errors to simulate real-world user input.

**Step 3: Dataset Splitting**
The final dataset will be split into three distinct sets to ensure proper model evaluation:
*   **Training Set (80%):** For training the models.
*   **Validation Set (10%):** For tuning model hyperparameters during training.
*   **Test Set (10%):** A held-out set, unseen by the model during training, to perform the final accuracy evaluation. This set will be carefully reviewed to include challenging and diverse examples.

## Phase 2: Solution Architecture

This phase outlines the technical components and the end-to-end workflow for processing a user's query. We will build a modular pipeline that is easy to test, maintain, and improve.

### 2.1. Model Selection

Based on the tasks defined in Phase 1, we will fine-tune two separate, specialized models:

1.  **Query Expansion Model (Seq2Seq):**
    *   **Model:** We will use a pre-trained sequence-to-sequence model like **T5 (Text-to-Text Transfer Transformer)**, specifically the `t5-small` or `t5-base` variant.
    *   **Why T5?** T5 is exceptionally well-suited for text generation tasks. It is trained on a multi-task objective, making it a powerful few-shot learner that can be easily fine-tuned to translate from a "contextual query" to a "standalone query."
    *   **Input Format:** `expand: {history} | query: {current_query}`
    *   **Output Format:** `{expanded_query}`

2.  **Topic Tagging Model (Classifier):**
    *   **Model:** We will use a pre-trained encoder-based model like **BERT (Bidirectional Encoder Representations from Transformers)**, specifically `bert-base-uncased`.
    *   **Why BERT?** BERT excels at understanding the nuances and context of text, making it a state-of-the-art choice for text classification. We will add a classification head on top of the BERT model to predict our hierarchical topics.
    *   **Input Format:** The `expanded_query` (output from the first model) will be fed into the classifier. This is crucial because the expanded query contains all the necessary context, making the classification task much more accurate and reliable.
    *   **Output Format:** A 2-level topic string, e.g., `Politics - India`.

### 2.2. The Inference Pipeline

The two models will be chained together in a simple, two-step pipeline for real-time inference:

```
User Query + Conversation History
           |
           v
+--------------------------+
|   1. T5 Expansion Model  |
| (Query Expansion)        |
+--------------------------+
           |
           v
     Expanded Query
           |
           v
+--------------------------+
|  2. BERT Classifier Model|
| (Topic Tagging)          |
+--------------------------+
           |
           v
     Expanded Query + Topic
```

**Workflow Steps:**

1.  **Input:** The system receives the user's `current_query` and the `history` of the last 10 dialogue exchanges (up to 20 messages).
2.  **Step 1 - Expansion:** The history and current query are formatted into a single string and passed to the fine-tuned T5 model. The model generates the `expanded_query`.
3.  **Step 2 - Tagging:** The `expanded_query` is then passed to the fine-tuned BERT model. The model classifies the text and outputs the most likely `topic`.
4.  **Output:** The final output is the `expanded_query` and its associated `topic`. This information can then be used by the downstream components of the chatbot to fetch and deliver the correct answer.

## Phase 3: Implementation and Evaluation Plan

This phase details the step-by-step process for building, training, and evaluating our models. We will use Google Colab for its free GPU resources, which are ideal for fine-tuning transformer models.

### 3.1. Environment Setup

*   **Platform:** Google Colab.
*   **Key Libraries:**
    *   `transformers`: For accessing pre-trained models (T5, BERT) and the fine-tuning APIs.
    *   `datasets`: For efficient data loading and processing.
    *   `torch`: The underlying deep learning framework.
    *   `scikit-learn`: For evaluation metrics.
    *   `pandas`: For data manipulation.

### 3.2. Implementation Steps

1.  **Data Preparation (Notebook 1):**
    *   Create the initial "gold" seed dataset of ~100 examples as a `.jsonl` file.
    *   Write a script to prompt an LLM (e.g., via an API) with few-shot examples from the seed set to generate a larger synthetic dataset (~2000 examples).
    *   Implement data augmentation techniques (paraphrasing, entity substitution).
    *   Combine and shuffle the data, then split into `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

2.  **Model 1 - T5 for Query Expansion (Notebook 2):**
    *   Load the datasets using the `datasets` library.
    *   Preprocess the data: format the `history` and `current_query` into the T5 input format (`expand: ...`).
    *   Load the pre-trained `t5-small` model and tokenizer.
    *   Set up the `Seq2SeqTrainer` from the `transformers` library.
    *   Fine-tune the model on the training set, using the validation set to monitor performance.
    *   Save the fine-tuned model and tokenizer for later use.

3.  **Model 2 - BERT for Topic Tagging (Notebook 3):**
    *   Load the datasets.
    *   Preprocess the data: use the `expanded_query` as the input and the `topic` as the label. Convert topic strings to integer labels.
    *   Load the pre-trained `bert-base-uncased` model and tokenizer.
    *   Add a classification head (`BertForSequenceClassification`).
    *   Set up the `Trainer`.
    *   Fine-tune the model.
    *   Save the fine-tuned model and tokenizer.

### 3.3. Evaluation Metrics (The Scientific Approach)

To meet the 95% accuracy target, we need to define and measure our performance rigorously.

*   **Query Expansion Model Evaluation:**
    *   **Primary Metric: ROUGE (Recall-Oriented Understudy for Gisting Evaluation).** This metric compares the model-generated expansion to the ground-truth expansion, measuring the overlap of n-grams. `ROUGE-L` (which measures the longest common subsequence) will be particularly important.
    *   **Secondary Metric: BLEU (Bilingual Evaluation Understudy).** While typically used for translation, it's also effective for measuring the precision of generated text against a reference.
    *   **Qualitative Assessment:** Manually review a sample of the test set predictions to identify common error patterns (e.g., incorrect entity resolution, incomplete expansions).

*   **Topic Tagging Model Evaluation:**
    *   **Primary Metric: Accuracy.** The percentage of correctly predicted topics on the test set. This directly addresses the evaluation criteria.
    *   **Secondary Metrics: Precision, Recall, and F1-Score (per-class).** This is crucial for understanding the model's performance on minority classes (less frequent topics). We will generate a classification report to analyze this.
    *   **Confusion Matrix:** To visualize which topics are being confused with each other. This can guide further data collection or model tuning.

### 3.4. Final Pipeline (Notebook 4):**

*   Load the fine-tuned T5 and BERT models.
*   Create a function that takes a `current_query` and `history` and implements the two-step pipeline.
*   Run this pipeline on the `test.jsonl` set.
*   Calculate the final evaluation metrics for both models and present the results in a clear, summary format.
*   Perform a final qualitative analysis of the end-to-end system's performance on the most challenging test cases.

## Phase 4: Future Improvements

Achieving a 95% accuracy is a great start, but pushing beyond that requires a structured approach to iterative improvement.

### 4.1. Error Analysis Driven Development

*   **Isolate Failure Modes:** After the initial evaluation, we will perform a deep dive on the test set failures. We'll categorize errors for both models (e.g., "Incorrect pronoun resolution," "Wrong topic for ambiguous entity," "Failed topic shift detection").
*   **Targeted Data Augmentation:** Based on the most common failure modes, we will generate or manually create new training data that specifically targets these weaknesses. For example, if the model struggles with topic shifts after a `GENERAL` turn, we will create more examples of that specific scenario.

### 4.2. Advanced Modeling Techniques

*   **Single, Multi-task Model:** Instead of two separate models, we could experiment with a single T5 model trained on both tasks simultaneously. The input could be prefixed with the task, e.g., `expand: ...` or `classify: ...`. This can sometimes lead to better performance as the model can learn shared representations.
*   **Larger Models:** If performance plateaus, we can experiment with larger base models (e.g., `t5-base` or `bert-large`). This provides more capacity to learn complex patterns but requires more computational resources.
*   **Parameter-Efficient Fine-Tuning (PEFT):** Techniques like LoRA (Low-Rank Adaptation) can be used to fine-tune larger models more efficiently, reducing training time and memory requirements.

### 4.3. Incorporating a Knowledge Base

*   **Entity Linking:** For more complex queries, the model's knowledge is limited to its training data. We could add an entity linking step to connect entities in the conversation (e.g., "UK") to a structured knowledge base like Wikidata. This would allow the system to:
    *   Disambiguate entities (e.g., "Georgia" the country vs. "Georgia" the US state).
    *   Access a richer set of facts to potentially improve topic classification.

### 4.4. Human-in-the-Loop Feedback

*   **Active Learning:** Set up a system where the model can flag low-confidence predictions for review by a human. This feedback can be used to continuously create high-quality training data, focusing on the examples the model finds most difficult. This is one of the most effective long-term strategies for improving model accuracy.
