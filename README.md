# Urdu Conversational Chatbot 🤖

This repository contains the code for an Urdu conversational chatbot built using a Transformer architecture implemented from scratch in PyTorch. The project aims to explore the capabilities of the Transformer model for sequence-to-sequence tasks in Urdu, based on the provided dataset.

## Project Overview

The primary goal was to build a custom conversational chatbot for Urdu using a Transformer encoder-decoder architecture *without* relying on pre-trained models. Key components like Multi-Head Attention, Positional Encoding, and Feed-Forward Networks were implemented from scratch. The project includes data preprocessing, model training, evaluation, and a simple web interface using Streamlit.

---

## Architecture 🏗️

The chatbot employs the standard **Transformer** model architecture:
* **Encoder:** Reads the input Urdu prompt and builds a contextual representation.
* **Decoder:** Takes the encoder's output and autoregressively generates the response, token by token.
* **Key Components (Built from Scratch):**
    * `nn.Embedding` Layer
    * `PositionalEncoding`
    * `MultiHeadAttention` (Scaled Dot-Product Attention)
    * `PositionwiseFeedForward` Network
    * `EncoderLayer` and `DecoderLayer` (combining attention and feed-forward with Layer Normalization and Residual Connections)
* **Framework:** PyTorch

---

## Dataset 📊

* **Source:** [Urdu Conversational Dataset (20,000 Sentences)](https://www.kaggle.com/datasets/muhammadahmedansari/urdu-dataset-20000) from Kaggle.
* **Challenge:** This dataset originates from the Common Voice project and consists primarily of **disconnected, non-conversational sentences**.
* **Pair Creation:** To adapt this for a chatbot, prompt-response pairs were created using a heuristic (pairing short sentences with the following sentence). This was necessary but introduced significant noise and illogical pairings into the training data.

---

## Preprocessing ⚙️

1.  **Loading:** Loaded only the `sentence` column from the `.tsv` file.
2.  **Normalization:** Applied robust text cleaning:
    * Removed punctuation (`۔`, `؟`, `,`, etc.).
    * Removed diacritics (zer, zabar, pesh).
    * Standardized variations of Alef (`آ` -> `ا`) and Yeh (`ی` -> `ی`).
    * Cleaned extra whitespace.
3.  **Pair Generation:** Created `(input_X, target_Y)` pairs using the heuristic mentioned above.
4.  **Tokenization:** Trained a custom **Unigram (subword)** tokenizer from scratch on the training corpus. Special tokens `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]` were added. The trained tokenizer was saved to `my_unigram_tokenizer.json`.
5.  **Splitting:** Divided the data into Training (80%), Validation (10%), and Test (10%) sets.
6.  **DataLoader:** Created PyTorch `Dataset` and `DataLoader` classes to handle tokenization, padding (`MAX_SEQ_LEN=100`), adding special tokens, and batching (`BATCH_SIZE=32`).

---

## Training 🏋️‍♂️

* **Hyperparameters:**
    * Embedding Dimension: 256
    * Layers (Encoder/Decoder): 2
    * Attention Heads: 2
    * FeedForward Dimension: 1024
    * Dropout: 0.3
    * Optimizer: Adam (Learning Rate: 0.0001, Weight Decay: 1e-5)
    * Loss Function: CrossEntropyLoss (ignoring PAD index)
* **Regularization:** Implemented techniques to combat initial overfitting observed with larger model dimensions:
    * Increased Dropout (0.1 -> 0.3)
    * Reduced Model Size (Embedding 512 -> 256)
    * Added Weight Decay to the optimizer.
* **Early Stopping:** Used validation loss with a patience of 3 to stop training when performance plateaued, automatically saving the best model state (`transformer_chatbot_best_model.pt`).
* **Mixed Precision:** Utilized `torch.cuda.amp` for faster training on GPU.

---

## Results & Conclusion 📉

* **Overfitting Solved:** Regularization techniques successfully prevented overfitting, as shown by the consistently decreasing validation loss during training.
* **Performance:** The model achieved relatively low accuracy (around 20-22% on validation) despite successful training.
* **Inference:** Generated responses are grammatically plausible Urdu but **contextually nonsensical** and unrelated to the prompts. The model often defaulted to repetitive answers.
* **Evaluation Metrics:** BLEU and ROUGE scores on the test set were **0.0**, quantitatively confirming the lack of meaningful overlap between generated and reference responses.
* **Conclusion:** The project successfully demonstrates the implementation of a Transformer from scratch. However, the final conversational quality is poor, **directly limited by the non-conversational nature and inherent noise of the provided dataset**. Even a correctly implemented model cannot create conversational logic if the training data lacks it.

---

## How to Run the App ▶️

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Ali-Hamza852/Urdu-Conversational-Chatbot.git](https://github.com/Ali-Hamza852/Urdu-Conversational-Chatbot.git)
    cd Urdu-Conversational-Chatbot
    ```
2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create
    python3 -m venv venv
    # Activate (Linux/macOS)
    source venv/bin/activate
    # Activate (Windows)
    .\venv\Scripts\activate
    ```
3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application should open in your web browser.

---

## Technology Stack 💻

* Python
* PyTorch
* Streamlit
* Pandas
* Custom Tokenizer (based on Unigram principles)
* Git / GitHub
