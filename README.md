# Reducing Hallucinations in LLMs

## Overview

This solution focuses on reducing hallucinations in large language models (LLMs) by implementing techniques that improve model reliability and trustworthiness. Hallucinations refer to instances when the LLM generates inaccurate or fabricated information. To address this, we employ strategies like **Grounded Generation using Retrieval-Augmented Generation (RAG)**, **Self-Consistency Checks**, and **LLM as a Judge**.

## Approach and Techniques

### Grounded Generation with Retrieval-Augmented Generation (RAG)
This technique integrates retrieved, verified information with LLM-generated responses. By using a retriever that fetches relevant documents based on a query, the model is constrained to produce responses aligned with validated content, reducing hallucination.

### Self-Consistency Check
The self-consistency check generates multiple responses to the same input query and cross-verifies the answers. Consistent responses across iterations reduce the likelihood of hallucinated information, as only those answers repeated across outputs are considered reliable.

### LLM as a Judge
In this technique, the model serves as its own evaluator, reviewing its responses against predefined criteria. By self-assessing response validity, the model can flag potentially inaccurate or speculative information, filtering out unreliable outputs.

## Module and Function Overview

### `AzureOpenAIModel` Class
The `AzureOpenAIModel` class initializes and configures the model. It allows setting parameters like temperature and verbosity to control the model’s response creativity and output logging.

- **Methods**:
    - **`__init__`**: Sets up the model configuration with parameters such as temperature and verbosity.
    - **`_create_prompt`**: Builds and returns a prompt template that combines system-level instructions and human-provided input.
    - **`run_model`**: Runs a simple model based on prompt-response with system and human inputs.
    - **`run_rag_model`**: Implements the core of Retrieval-Augmented Generation. It combines retrieved documents with prompts, constraining the model to respond using factual information.

## RAG Implementation Details

In this example, we use Wikipedia data to answer a factual question about a specific entity ("KP Sharma Oli"), demonstrating how RAG minimizes hallucinations by anchoring responses to reliable sources.

### Document Loading and Preprocessing

1. **Data Loading**:
   - The **`WikipediaLoader`** loads documents related to a specified topic. Here, the query `"KP Sharma Oli"` is used to fetch up to 2 documents about this subject from Wikipedia.
   - This step is critical because it initializes the RAG process with trustworthy, up-to-date information, forming the knowledge base for model responses.

   ```python
   # Load Wikipedia data
   docs = WikipediaLoader(query="KP Sharma Oli", load_max_docs=2).load()```

2. **Text Splitting**
- The **`RecursiveCharacterTextSplitter`** divides each document into smaller, overlapping chunks (500 characters with a 50-character overlap).
- These chunks improve retrieval accuracy by ensuring that the retriever has focused, manageable portions of information to search, which is especially useful for long documents.
- Chunks also prevent the model from missing critical context in long or complex answers, as the retriever can pull only the relevant portions.

```python
# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
```

3.  **Embedding Creation and Vector Storage**

- Using AzureOpenAIEmbeddings, each document chunk is converted into an embedding, a dense vector representation of text that captures semantic meaning.
- This embedding generation step uses OpenAI’s Azure service, with the model parameters specified in environment variables for secure access.
- Embeddings allow for similarity-based retrieval, enabling the system to find text chunks closely aligned with the user’s query.

```python
# Create a vector store from the embeddings and store it in Chroma database
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory="./chroma_db"
)

```

### Example Workflow

Given a question, **"When was the last time KP Sharma Oli was elected as the Prime Minister of Nepal?"**, the following steps illustrate the RAG implementation:

1. **Define the Human Prompt**:
   The prompt instructs the model to generate a response based on retrieved context and to abstain from answering if information is lacking. This helps avoid fabrications and ensures concise responses.

    ```plaintext
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    ```

2. **Execution of RAG**:
   - The `run_rag_model` method runs the prompt with the retriever to fetch the required context from the vector database.
   - The model combines the retrieved context with the human prompt to produce an answer.
   
3. **Output with and without RAG**:
   The model outputs:

- Without RAG: ```KP Sharma Oli was last elected as the Prime Minister of Nepal on February 15, 2018.```
- With RAG: ```The last time KP Sharma Oli was elected as the Prime Minister of Nepal was on 15 July 2024.```

### Handling Edge Cases and Failure Modes

To further improve robustness and accuracy:
- **Fallback Mechanism**: When the retriever fails to find adequate context, the model is programmed to abstain from answering or flag uncertainty.
- **Fine Tuning**: Instruction fine tuning or model fine tuning can be incorporated.

## Self Consistency Implementation Details

### Prompt Design

A prompt (`human_prompt`) is crafted to guide the model through the problem and specify the desired output format. It requires the LLM to provide a response in JSON format with two keys:
- **`analysis`**: Contains an explanation or analysis of the solution.
- **`result`**: Contains the integer representing the final answer.

Example prompt:

```plaintext
John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How much lego sets does he still have?

Output formatting: Output the result in json format. The json object should have two keys "analysis" and "result", with the value being the number of lego sets John still has. For example:
{
    "analysis": // any analysis you want to provide
    "result": // the number of lego sets John still has in integer
}
```

Result from 10 iterations:

```
[2, 2, 11, 2, 2, 11, 2, 2, 2, 2]
```
Final result:
```
2
```

## LLM as a Judge Implementation Details

### Prompt Design

The prompt for the language model is structured to ensure detailed assessment along the following dimensions:

1. **Key Information Coverage**: Evaluates if the summary includes main points and critical details accurately.
2. **Logical Flow and Structure**: Assesses whether the ideas in the summary are organized logically, reflecting the flow of the original text.
3. **Factual Accuracy**: Checks if all statements in the summary are factually correct, with no misrepresentations or unsupported claims.
4. **Conciseness and Relevance**: Rates the summary’s conciseness, ensuring it contains only relevant details without verbosity.
5. **Terminology and Tone Consistency**: Ensures that terminology and tone are consistent with the original text, with no added bias or stylistic mismatch.

Each criterion is evaluated as either `true` or `false`, providing a straightforward yet comprehensive score to measure the coherence of the summary against the original article.

### Example Prompt Structure

The following example shows the prompt structure that is used to guide the language model’s evaluation:

```plaintext
You will be given a summary (LLM output) written for a news article and asked to evaluate its coherence compared to the original text. 
Please assess the summary based on five specific criteria, providing a true or false rating for each one, and output the results in JSON format.

### Criteria for Evaluation:
1. **Key Information Coverage**: Does the summary accurately capture the main points and critical details from the original text?
2. **Logical Flow and Structure**: Is the summary logically structured, with ideas presented in a clear and cohesive order, mirroring the logical flow of the original text?
3. **Factual Accuracy**: Are all factual statements in the summary accurate based on the original text, with no misrepresentations or unsupported claims?
4. **Conciseness and Relevance**: Is the summary concise, focusing on relevant details without unnecessary information or excessive verbosity?
5. **Terminology and Tone Consistency**: Does the summary use terminology and maintain a tone consistent with the original text, ensuring no added bias or style mismatch?

### Format:
Output your evaluation in JSON format, with each criterion labeled and rated as either true or false.

Original Text:
{input}

Summary:
{llm_output}

Evaluation:
{
    "Key Information Coverage": true/false,
    "Logical Flow and Structure": true/false,
    "Factual Accuracy": true/false,
    "Conciseness and Relevance": true/false,
    "Terminology and Tone Consistency": true/false
}

```

Final result in the implemented example in jupyter notebook:
```
{'Key Information Coverage': True, 'Logical Flow and Structure': True, 'Factual Accuracy': True, 'Conciseness and Relevance': True, 'Terminology and Tone Consistency': True}
Score of this LLM output: 5/5
```

# Speech-To-Text To Reduce Ghost Results Overview
This repository provides a implementation to reduce "ghost results" (inaccurate or fabricated transcriptions) in a speech-to-text (STT) system using techniques such as noise reduction and silence trimming. The solution leverages the Whisper model and various preprocessing techniques to improve transcription accuracy and reliability. Also, postprocessing using a language model is implemented.

# Implementation Details

This section outlines the key components of the speech-to-text (STT) system that reduces ghost results through various preprocessing and post-processing techniques.

## 1. Loading the Model and Processor

The system starts by loading the Whisper model and its associated processor. This process includes configuring the model with the appropriate model name, specifying the cache directory for storing model weights, and selecting the device (CPU or GPU) on which the model will operate. By loading the model in this manner, the system can effectively handle audio data for transcription.

## 2. Silence Detection and Trimming

To enhance transcription accuracy, the system incorporates a silence detection technique. This involves analyzing the audio file to identify the duration of silence at the beginning of the audio segment. By determining when the actual speech starts, the system trims any leading silence from the audio. This reduces the risk of misinterpretation caused by initial silence, which can contribute to ghost results in transcriptions.

## 3. Noise Reduction

Another technique implemented in the system is noise reduction, aimed at improving the clarity of the audio signal. By applying spectral gating, background noise is filtered out from the audio. This enhancement ensures that the transcription model can concentrate on the relevant speech content, leading to more accurate and reliable transcription results. 

## 4. Audio Resampling

To maintain compatibility with the Whisper model, the audio files undergo resampling to a target sample rate. This step verifies if the original sample rate matches the desired rate; if they differ, the audio is resampled. This uniformity ensures that the audio data remains intact during the transcription process.

## 5. Transcription

After preprocessing the audio (via silence trimming, noise reduction, or resampling), the transcription process is initiated. The Whisper model is employed to convert the processed audio into text. During this step, the processor prepares the audio features, which are subsequently input into the model, generating a predicted transcription.

## 6. Post-Processing with Language Model

Following the initial transcription, post-processing is performed using a large language model (LLM) from GPT. This involves providing a human-readable prompt that requests corrections to the raw transcription output. The instructions include correcting misheard words and phrases, ensuring proper grammar, punctuation, and sentence structure, and maintaining the original meaning and tone.
