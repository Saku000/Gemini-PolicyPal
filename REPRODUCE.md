# REPRODUCE.md

## 1. Project Overview

This project, **Gemini-PolicyPal**, is an AI-powered insurance policy analysis system built using:

* Google Gemini API (LLM + Embeddings)
* Retrieval-Augmented Generation (RAG)
* Streamlit (frontend UI)

It supports:

* Policy document ingestion (PDF)
* Automated policy analysis
* Question answering (RAG-based)
* Policy comparison with structured outputs and visualizations

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
git clone https://github.com/Saku000/Gemini-PolicyPal.git
cd Gemini-PolicyPal
```

---

### 2.2 Create Virtual Environment

#### Windows (Recommended)

```bash
setup.bat
```

#### Manual Setup (All Platforms)

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. API Key Configuration

This project requires a **Google Gemini API Key**.

### Step 1: Create `.env` file in root directory

```bash
GEMINI_API_KEY=your_api_key_here
```

### Step 2: Replace with your actual API key

You can obtain a key from:
https://ai.google.dev/

---

## 4. Data Preparation

### 4.1 Folder Structure

Ensure the following folders exist:

```
data/
├── qa_policies/
├── compare_prod/
│   ├── policy_a/
│   └── policy_b/
```

---

### 4.2 Add Policy Documents

Place your PDF insurance policies:

* QA Mode:

```
data/qa_policies/
```

* Compare Mode:

```
data/compare_prod/policy_a/
data/compare_prod/policy_b/
```

---

## 5. Running the Application

### Option 1: Using Script (Windows)

```bash
run.bat
```

### Option 2: Manual Run

```bash
streamlit run app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 6. How to Reproduce Results

### 6.1 Dashboard (Auto Analysis)

* Upload or place a policy PDF in `data/qa_policies/`
* Navigate to **Dashboard**
* The system will:

  * Parse PDF
  * Extract structured insights
  * Generate risk score and summary

---

### 6.2 Ask Pal (RAG QA)

* Go to **Ask Pal**
* Ask questions like:

  * "Is vandalism covered?"
  * "What is the deductible?"
* System retrieves relevant chunks and answers using Gemini

---

### 6.3 Compare Policies

* Place two policies:

  * `policy_a/`
  * `policy_b/`
* Go to **Compare**
* System will:

  * Build vector index for both policies
  * Retrieve evidence
  * Generate structured comparison
  * Display radar chart

---

## 7. Output / Results

The system produces:

### 7.1 Structured JSON Analysis

* Policy metadata
* Coverage details
* Risk flags
* Risk score

### 7.2 RAG Answers

* Context-grounded responses
* Evidence-based reasoning

### 7.3 Comparison Results

* Category-wise scores
* Winner determination
* Trade-offs
* Radar visualization

---

## 8. Reproducibility Notes

* Results may slightly vary due to:

  * LLM stochasticity (Gemini)
  * API updates
* To improve consistency:

  * Keep the same model version (`gemini-2.5-flash`)
  * Use identical input PDFs

---

## 9. Dependencies

Main libraries:

* google-genai
* streamlit
* pdfplumber
* numpy / scipy
* rank-bm25
* plotly
* tiktoken

Install all via:

```bash
pip install -r requirements.txt
```

---

## 10. Troubleshooting

### Issue 1: API Key Error

* Ensure `.env` exists
* Ensure key name is exactly:

```
GEMINI_API_KEY
```

---

### Issue 2: No Results / Empty Output

* Check PDF is readable (not scanned image)
* Ensure PDF is placed in correct folder

---

### Issue 3: Streamlit Not Running

```bash
pip install streamlit
```

---

## 11. Expected Runtime

* Initial indexing: ~5–15 seconds per policy
* Query response: ~1–3 seconds
* Comparison: ~10–20 seconds

---

## 12. Reproducibility Checklist

✔ Clone repo
✔ Install dependencies
✔ Add API key
✔ Add PDF files
✔ Run Streamlit app
✔ Execute Dashboard / QA / Compare

---

## 13. Contact

If reproduction fails, please check:

* File paths
* API key validity
* Dependency installation

---
