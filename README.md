# Shakespeare RAG Chatbot

## 1. Project Overview

This system lets you ask questions about three Shakespeare plays and get answers grounded in the actual text. It uses Retrieval-Augmented Generation (RAG): for every question, it first finds the most relevant scene passages from the plays, then passes those passages to a small language model (SLM) so the answer is based on evidence rather than guesswork.

The three plays covered are **Hamlet**, **Macbeth**, and **Romeo and Juliet**.

The language model is **qwen2.5:1.5b**, a 1.5 billion parameter model that runs fully locally on your CPU via [Ollama](https://ollama.com). No internet connection, no API key, and no GPU are needed at inference time.

---

## 2. System Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running on your machine
- At least 4 GB of free disk space (for the model, embeddings, and dependencies)
- No GPU required -- the system runs on a standard CPU laptop

---

## 3. Installation

### Step 1 -- Clone the repository

Download the project to your local machine.

```bash
git clone https://github.com/JeetCoD/ml-final
```

### Step 2 -- Navigate into the project folder

Move into the `ml` subdirectory where the source code lives.

```bash
cd ml-final/ml
```

### Step 3 -- Install Python dependencies

This installs all required libraries listed in the requirements file.

```bash
pip install -r requirements.txt
```

### Step 4 -- Install the Ollama client and Keras fix

Run this command to install the Ollama Python client (used to talk to the local language model) and `tf-keras`, which fixes a Windows compatibility issue where Keras 3 is loaded instead of the older version that `sentence-transformers` expects.

```bash
pip install ollama tf-keras
```

### Step 5 -- Pull the language model

This downloads the qwen2.5:1.5b model to your machine via Ollama. Make sure the Ollama app is open and running before running this command.

```bash
ollama pull qwen2.5:1.5b
```

---

## 4. Dataset Setup

The three play JSON files must be placed inside the `shakespeare_slm_dataset/` folder at the root of the `ml/` directory. The folder should look like this:

```
ml/
  shakespeare_slm_dataset/
    hamlet.json
    macbeth.json
    romeo_and_juliet.json
```

These files are provided with the submission. If the folder does not exist yet, create it and copy the three files in.

---

## 5. Build the Index (Run Once)

Before starting the chatbot you need to build the retrieval index. This script loads all three plays, splits them into scene-level chunks, encodes every chunk into an embedding vector using a sentence transformer model, and saves the results to disk.

```bash
python src/build_index.py
```

**This takes about 1 to 2 minutes on first run** because encoding all the chunks on CPU takes time. You only need to run it once. After it finishes, the following files are saved so the chatbot can load them instantly on every future startup:

- `results/embeddings.npy` -- the embedding matrix (one vector per scene chunk)
- `results/chunks.json` -- the scene chunk data with metadata

---

## 6. Run the Chatbot

Start the interactive question-answering chatbot with this command:

```bash
python src/rag_chatbot.py
```

The chatbot has two modes:

- **Normal mode** -- type any question about the plays and the system will retrieve relevant scenes and generate a grounded answer.
- **Stylised mode** -- type `stylised:` before your question to get a short creative response written in Shakespearean English.

### Example questions

```
Why does Macbeth kill Duncan?
```

```
Who is Hamlet?
```

```
What is the conflict between the Montagues and the Capulets?
```

```
stylised: Write a response from Juliet about her love for Romeo
```

Type `quit` or `exit` to stop the chatbot.

---

## 7. Run the Baseline System

The baseline system answers questions using only the system prompt and the question itself. It does not retrieve any Shakespeare passages. This is used to compare against the RAG system to show how much retrieval improves the answers.

```bash
python src/baseline.py
```

---

## 8. Run the Evaluation

This script runs all evaluation questions through both the baseline system and the RAG system, then saves every question-answer pair to a CSV file for manual scoring.

```bash
python src/evaluate.py
```

Results are saved to:

```
results/evaluation_results.csv
```

If you want to check the CSV structure without calling the language model (for example, to test that retrieval works without waiting for generation), use mock mode:

```bash
python src/evaluate.py --mock
```

---

## 9. Project File Structure

```
ml/
  src/
    config.py           - all project settings in one place
    data_loader.py      - loads the three play JSON files
    chunking.py         - splits scenes into retrieval chunks
    retrieval.py        - embedding model and cosine similarity search
    build_index.py      - run this once to build and save the index
    rag_chatbot.py      - the main interactive chatbot
    baseline.py         - prompt-only baseline for comparison
    evaluate.py         - runs evaluation questions through both systems
    dataset_stats.py    - computes and saves dataset statistics

  shakespeare_slm_dataset/
    hamlet.json
    macbeth.json
    romeo_and_juliet.json

  prompts/
    system_prompt.txt                  - instructions given to the language model
    stylised_generation_prompt.txt     - prompt template for creative responses

  results/
    embeddings.npy          - saved embedding matrix (created by build_index.py)
    chunks.json             - saved scene chunks (created by build_index.py)
    evaluation_results.csv  - evaluation output (created by evaluate.py)
```

---

## 10. Troubleshooting

**`No module named ollama`**

Install the Ollama Python client.

```bash
pip install ollama
```

---

**`No module named tf_keras`**

Install the Keras compatibility fix.

```bash
pip install tf-keras
```

---

**Build index stops silently at step 3 with no error**

This is a Windows DLL conflict. TensorFlow (loaded by `sentence-transformers`) and NumPy's OpenBLAS both try to register incompatible BLAS symbols, which causes a silent crash. The fix is already in place in `retrieval.py`: `sentence_transformers` is imported before `numpy` so TensorFlow loads first and claims the symbols. If you see this problem, make sure you have not reordered the imports.

---

**`Error: model 'qwen2.5:1.5b' not found`**

The model has not been downloaded yet, or the Ollama app is not running. Open the Ollama app first, then run:

```bash
ollama pull qwen2.5:1.5b
```

---

**Encoding is very slow**

This is normal on CPU. Encoding all scene chunks takes 1 to 2 minutes the first time you run `build_index.py`. After that the index is cached to disk and the chatbot loads it in seconds. You will not see this delay again unless you delete the cache files.

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Storing and computing with embedding vectors |
| `pandas` | Writing statistics and evaluation results to CSV |
| `scikit-learn` | Cosine similarity calculation |
| `sentence-transformers` | Encoding text into embedding vectors |
| `tqdm` | Progress bar during encoding |
| `matplotlib` | Generating the chunk count bar chart |
| `ollama` | Python client for the local Ollama language model |
| `tf-keras` | Windows fix for Keras 3 compatibility with sentence-transformers |

---

# Demo question for presentation
1. Who is Hamlet?
2. Why does Macbeth kill Duncan?
3. What is the conflict between the Montagues and the Capulets?
4. stylised: Write a response from Juliet about her love for Romeo


Built for CSCI433/933 Machine Learning Algorithms and Applications, Assignment 2, University of Wollongong.
