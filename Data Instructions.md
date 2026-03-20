# Dataset / Data Instructions

## 1. Data Description

This project does not rely on a fixed structured dataset.
Instead, it uses **insurance policy documents (PDF format)** as input data.

These documents serve as the knowledge base for:

* Policy analysis
* Question answering (RAG)
* Policy comparison

---

## 2. Data Source

Users can use:

* Publicly available insurance policy samples
* Sample documents from insurance company websites
* Self-provided policy PDFs

Example sources:

* State Farm sample policies
* Allstate policy documents
* Any publicly accessible insurance PDFs

---

## 3. How to Use Data

### 3.1 QA / Analysis Mode

Place PDF files into:

```
data/qa_policies/
```

Example:

```
data/qa_policies/sample_policy.pdf
```

---

### 3.2 Comparison Mode

Place two policies into:

```
data/compare_prod/policy_a/
data/compare_prod/policy_b/
```

Example:

```
data/compare_prod/policy_a/policy_a.pdf
data/compare_prod/policy_b/policy_b.pdf
```

---

## 4. Notes

* PDFs should be **text-based (not scanned images)** for best results
* Larger policies may take longer to process
* Different documents will produce different analysis results

---

## 5. Optional (For Reproducibility)

For consistent reproduction, users can use the same sample policies provided in:

```
data/sample_policies/
```

(if included in the repository)

---

## 6. Why No Fixed Dataset

This system is designed as a **general-purpose policy analysis tool**,
so it works with arbitrary insurance documents instead of a predefined dataset.
