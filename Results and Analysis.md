# Results and Analysis

## 1. Overview

This project evaluates the performance of **Gemini-PolicyPal**, an AI-powered system for insurance policy understanding using Retrieval-Augmented Generation (RAG).

The evaluation focuses on three main functionalities:

1. Automated policy analysis
2. Question answering (RAG-based)
3. Policy comparison

---

## 2. Experimental Setup

### Input Data

* Insurance policy PDFs (auto / home insurance)
* Documents placed in:

  * `data/qa_policies/`
  * `data/compare_prod/`

### Model Configuration

* LLM: `gemini-2.5-flash`
* Embedding Model: `gemini-embedding-001`
* Retrieval:

  * Dense embeddings
  * BM25 hybrid retrieval (for comparison module)

---

## 3. Results

### 3.1 Automated Policy Analysis

The system successfully extracts structured information from unstructured PDFs, including:

* Policy type
* Coverage limits
* Deductibles
* Exclusions
* Risk score

#### Example Output

* Risk Score: **6.5 / 10**
* Key Risk Flags:

  * High deductible
  * Limited coverage on vandalism
  * Missing claim condition details

#### Analysis

* The model demonstrates strong ability to **summarize complex legal text**
* JSON-constrained prompting improves output consistency
* However, performance depends on **document clarity and structure**

---

### 3.2 Question Answering (RAG)

The RAG system retrieves relevant chunks and generates context-grounded answers.

#### Example Queries

| Question                | Result                          |
| ----------------------- | ------------------------------- |
| Is vandalism covered?   | Correctly identified as covered |
| What is the deductible? | Extracted correct value         |
| Are there exclusions?   | Listed relevant exclusions      |

#### Analysis

* Retrieval significantly reduces hallucination
* Multi-chunk context improves answer completeness
* Source grounding ensures **traceability**

Limitations:

* Performance drops if:

  * PDF text extraction is noisy
  * Key info is scattered across pages

---

### 3.3 Policy Comparison

The system compares two policies across multiple dimensions:

* Coverage Completeness
* Affordability
* Flexibility
* Exclusion Risk
* Claims Process
* Overall Value

#### Example Output

* Policy A:

  * Strong coverage
  * Higher premium

* Policy B:

  * Lower cost
  * More exclusions

* Overall Winner: **Policy A**

#### Visualization

* Radar chart clearly shows trade-offs between policies

#### Analysis

* Hybrid retrieval (dense + BM25) improves evidence quality
* Structured comparison avoids vague LLM outputs
* Missing-value detection prevents misleading conclusions

---

## 4. Key Strengths

### 4.1 Robust RAG Pipeline

* Combines embedding retrieval + LLM reasoning
* Reduces hallucination
* Supports explainable outputs

### 4.2 Structured Output Design

* JSON-based outputs improve reliability
* Enables downstream visualization and comparison

### 4.3 Anti-Hallucination Mechanisms

* Evidence retrieval before generation
* Placeholder detection (`$000`, `TBD`, etc.)
* Avoids making unsupported claims

---

## 5. Limitations

### 5.1 Dependence on PDF Quality

* Scanned PDFs or poor formatting reduce accuracy

### 5.2 LLM Variability

* Outputs may vary slightly across runs
* Not fully deterministic

### 5.3 Limited Numerical Precision

* Complex financial values may not always be perfectly extracted

---

## 6. Future Improvements

* Add OCR support for scanned PDFs
* Fine-tune retrieval ranking
* Add evaluation metrics (precision/recall for QA)
* Support more policy types (health, life insurance)

---

## 7. Conclusion

Gemini-PolicyPal demonstrates that combining:

* LLM reasoning
* Retrieval systems
* Structured prompting

can effectively transform complex insurance documents into:

* Actionable insights
* Explainable answers
* Structured comparisons

This system shows strong potential for real-world applications in:

* Insurance advisory
* Risk assessment
* Financial decision-making

---
