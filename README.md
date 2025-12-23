#  Climate Awareness Classification Using AI

## Overview
This project applies machine learning and natural language processing (NLP) to analyze public awareness of climate change using Twitter data. The aim is to classify climate-related tweets into different awareness categories, supporting research on public perception and sustainability-focused communication.

The project emphasizes **interpretable and responsible AI**, making it suitable for societal and policy-oriented analysis rather than automated decision-making.

---

## Problem Statement
Understanding public awareness and attitudes toward climate change is essential for designing effective sustainability policies and communication strategies. Social media platforms generate large volumes of climate-related content, making manual analysis difficult.

This project explores how machine learning can be used to classify tweets based on their stance toward climate change, helping to identify trends in public discourse.

---

## Dataset
The dataset contains **43,943 climate-related tweets** collected between **April 2015 and February 2018**.

- Each tweet was labeled independently by **three human annotators**
- Only tweets with full agreement were included
- This ensures high annotation reliability

  ---
  ### Dataset Source
The dataset is publicly available and can be accessed here:  
🔗 https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment

### Awareness Classes 
- **2 – News**: Factual or informational climate-related news  
- **1 – Pro**: Supports the belief in human-caused climate change  
- **0 – Neutral**: Neither supports nor refutes human-caused climate change  
- **-1 – Anti**: Denies or dismisses human-caused climate change  

---

## Methodology
The project follows a classical NLP pipeline:

1. Text cleaning and preprocessing  
2. Feature extraction using **TF-IDF** (unigrams and bigrams)  
3. Multi-class classification using **Logistic Regression**  
4. Model evaluation using accuracy, precision, recall, and confusion matrix  

This approach was selected to balance **performance, transparency, and computational efficiency**.

---

## Results
The model performs well on:
- Explicit supportive statements
- Neutral and factual news content

Challenges were observed in:
- Sarcastic or dismissive language
- Implicit skepticism without explicit denial terms

These limitations are consistent with known constraints of bag-of-words–based NLP models.

---

## Ethical Considerations
- Social media data may not represent all population groups
- Misclassification of opinions can introduce bias
- Results should be used for **analysis and awareness**, not automated policy decisions

Transparency and acknowledgment of limitations are critical when applying AI to societal issues.

---

## Sustainability Impact
By enabling scalable analysis of climate-related discourse, this project supports:
- Climate awareness research
- Responsible sustainability communication
- Ethical use of AI in social contexts

---

## Technologies Used
- Python  
- pandas  
- scikit-learn  
- TF-IDF Vectorization  
- Logistic Regression  

---

## Limitations and Future Work
- Limited handling of sarcasm and contextual nuance  
- Class imbalance across awareness categories  
- Future work may explore:
  - Context-aware language models
  - Bias mitigation techniques
  - Longitudinal analysis of awareness trends  


