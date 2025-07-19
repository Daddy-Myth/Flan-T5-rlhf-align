# Fine-tuning FLAN-T5 with Reinforcement Learning from Human Feedback (RLHF) for Neutral, Grammatically Correct News Summaries

## Project Overview
This project fine-tunes the `google/flan-t5-base` model using **Reinforcement Learning from Human Feedback (RLHF)** via Proximal Policy Optimization (PPO) to generate **news summaries** that are both **grammatically correct** and **neutral in sentiment**.

We use **two pretrained RoBERTa models as reward functions** to guide the alignment:

- ✅ `textattack/roberta-base-CoLA` — for **CoLA-based grammatical acceptability**
- ✅ `cardiffnlp/twitter-roberta-base-sentiment` — for **sentiment neutrality**

During PPO training, summaries that are both grammatically sound and neutral are rewarded, steering the summarizer towards more aligned, human-preferred outputs.

🧪 Evaluation is performed using:
- A real-world news summarization dataset: [`argilla/news-summary`](https://huggingface.co/datasets/argilla/news-summary)
- Batch scoring with both reward models
- Behavioral comparison with the base `flan-t5-base` model

📊 The full training pipeline uses:
- Hugging Face’s `transformers` + `trl` for RLHF
- 🤗 Model card hosting & tracking via Hugging Face Hub
- Weights & Biases (W&B) for logging and visualization (optional)

> 🎯 Goal: Teach the model to summarize **without bias** and **with clarity** — the way a human editor would prefer.
---

## ⚙️ Installation

```bash
Copy
Edit
# Clone the repository
git clone https://github.com/Daddy-Myth/Flan-T5-rlhf-align.git
cd Flan-T5-rlhf-align

# Create and activate a conda environment
conda create -n rlhf python=3.10 -y
conda activate rlhf

# Install required Python packages
pip install -r requirements.txt

# (Optional) Install Jupyter and ipywidgets for running notebooks
pip install notebook ipywidgets
```
✅ GPU Users: Make sure you have the correct PyTorch version for your CUDA setup. For CUDA 11.8:

```bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
💡 You may also want to install Git LFS for handling large model files:

```bash
Copy
Edit
# Ubuntu
sudo apt install git-lfs

# macOS (Homebrew)
brew install git-lfs

# Windows (Chocolatey)
choco install git-lfs

# Initialize LFS in the repo
git lfs install
```

---

## 🎯 RLHF Training Overview

The core of this project is an **RLHF (Reinforcement Learning from Human Feedback)** loop, implemented using Hugging Face's `trl` library with **Proximal Policy Optimization (PPO)**. The model learns to generate summaries that are both grammatically correct and sentiment-neutral.

### 🧠 Step-by-Step Pipeline

1. **Pretrained Base Model**
   - Use `google/flan-t5-base` as the base text-to-text model.
   - Load it into a PPOTrainer wrapper for fine-tuning.

2. **Prompt Sampling**
   - Use articles from the [`argilla/news-summary`](https://huggingface.co/datasets/argilla/news-summary) dataset.
   - Format each example as:  
     ```
     summarize: <news article>
     ```

3. **Generate Summaries**
   - The model generates a summary for each sampled article.

4. **Reward Scoring**
   - Each generated summary is scored using **two RoBERTa-based reward models**:
     - `textattack/roberta-base-CoLA`: evaluates **grammatical acceptability**.
     - `cardiffnlp/twitter-roberta-base-sentiment`: evaluates **sentiment neutrality**.
   - Rewards are scaled and combined.

5. **Policy Update with PPO**
   - PPO uses the reward signal to update the policy (i.e., the model’s parameters).
   - The goal is to increase the likelihood of grammatically correct and neutral summaries.

6. **Logging & Tracking**
   - Training logs and evaluation metrics are optionally tracked via [Weights & Biases](https://wandb.ai/).
   - Aligned models and tokenizer are pushed to the [Hugging Face Hub](https://huggingface.co/profoz/t5-aligned-summaries).

---

> ✅ After training, the aligned summarizer is compared against the base `flan-t5-base` model using batch evaluation on unseen news articles.

---

##  Example Usage

After training, the aligned model can be used via Hugging Face's `pipeline` interface:

<img width="2307" height="608" alt="image" src="https://github.com/user-attachments/assets/92689065-2505-4567-a07e-d8a398311753" />

---

## 📈 Evaluation

We compare the aligned model (`FLAN-T5 after RLHF`) to the original `flan-t5-base` using two reward metrics:

- **Neutral Sentiment Reward** using `cardiffnlp/twitter-roberta-base-sentiment`
- **Grammatical Acceptability (CoLA Score)** using `textattack/roberta-base-CoLA`

Evaluation was done on **1409 samples** from the [`argilla/news-summary`](https://huggingface.co/datasets/argilla/news-summary) dataset.

### 🔬 Quantitative Results

| Metric                      | FLAN-T5 (Before RLHF) | FLAN-T5 (After RLHF) |
|----------------------------|------------------------|-----------------------|
| Average Neutral Reward     | 1.3033                 | **1.3189**            |
| Average CoLA Reward        | 0.7961                 | **0.8650**            |
| Median Neutral Reward      | ~1.28                  | **~1.30**             |
| Median CoLA Reward         | ~0.87                  | **~0.97**             |

> 🧠 **TL;DR**: After RLHF training, FLAN-T5 generates summaries that are **more neutral in sentiment** and **significantly better in grammar**, as shown by both mean and median reward improvements.

---

### 📷 Visual Comparison

Below is a visualization of the average and median reward scores before and after alignment:

![Comparison of old vs new summarizer rewards](https://github.com/user-attachments/assets/c7cdbce1-5097-442c-91c0-c488dd7747a0)

- **Left**: Sentiment (Neutrality) — slight improvement in average & median  
- **Right**: Grammar (CoLA) — significant boost post-RLHF

---

##  Acknowledgments

- 📚 [Quick Start Guide to LLMs](https://github.com/sinanuozdemir/quick-start-guide-to-llms) — the course that inspired and guided this project.
- 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers) — for powerful model APIs and pipeline tools.
- 🧠 [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — for the PPOTrainer and RLHF infrastructure.
- 📰 [Argilla News Summary Dataset](https://huggingface.co/datasets/argilla/news-summary) — used for evaluation and benchmarking.
- 📈 [Weights & Biases](https://wandb.ai/) — for experiment tracking and visualization.

  
