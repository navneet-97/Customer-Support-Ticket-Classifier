
# 🎫 Customer Support Ticket Classifier

A Machine Learning pipeline that classifies customer support tickets by **Issue Type** and **Urgency Level** using Natural Language Processing (NLP). This project features an interactive Gradio web interface and is deployable to Hugging Face Spaces.

---

🎥 **Demo Video**: [Watch here](https://www.loom.com/share/6f853478bc624af88b8ff668242386a4?sid=db61d37c-6a9e-409a-a5cb-c7fe365761a5)

---

## 📌 Features

- 🧹 Text preprocessing using NLTK and TextBlob
- 🧠 Feature extraction via TF-IDF and engineered numerical features
- 🔤 Label encoding using `LabelEncoder`
- 🤖 Classification using Logistic Regression, Random Forest, and SVM
- 🌐 Interactive UI with Gradio
- 🚀 Deployable to [Hugging Face Spaces](https://huggingface.co/spaces)

---

## ⚙️ Setup Instructions

### ✅ Step 1: Create and Activate Virtual Environment

```bash
python -m venv ticket_classifier_env
```

#### Activate the environment:

- On **Windows**:
  ```bash
  ticket_classifier_env\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source ticket_classifier_env/bin/activate
  ```

---

### ✅ Step 2: Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Then, install NLTK corpora used in preprocessing:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

---

### ✅ Step 3: Prepare Dataset

Ensure that the dataset file `ai_dev_assignment_tickets_complex_1000.xls` is placed in the root project directory.

The dataset must contain the following columns:

- `ticket_id`
- `ticket_text`
- `issue_type`
- `urgency_level`
- `product`

---

### ✅ Step 4: Run Locally

Run the classifier locally to test:

```bash
python ticket_classification_pipeline.py
```

A browser window will open with the interactive Gradio UI where you can input support ticket texts and receive predictions for issue type and urgency level.

---
