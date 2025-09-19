# Clinical-Copilot
Clinical Copilot transforms raw patient data into clear, evidence-backed insights. It filters noise, retrieves guidelines, and delivers prioritized differentials, diagnostics, treatments, and red-flag alerts. Built with FastAPI + Tailwind, it empowers doctors to act faster, smarter, and with confidence.

# Features
-- *FHIR Parsing*: Ingests structured patient data bundles (FHIR JSON).  
- *Knowledge Retriever*: Retrieves relevant chunks from guideline documents (Jaccard similarity).  
- *Tiny GPT Inference*: Option to plug in a small causal LM for JSON-based reasoning.  
- *Rule Fallback*: Deterministic rules ensure suggestions even if ML is unavailable.  
- *Evidence Linking*: Outputs citations from guideline files.
- *UI*: Frontend built with Tailwind + vanilla JS.

# Project-Structure
clinical_copilot
|-app/
| |-main.py
| |-retriever.py
| |-model_infer.py
| |-rules_fallback.py
| |-insights_module.py
| |-knowledge/
|
|-fhir_db.json
|-static/
| |-landing.html
| |-app.html
|-requirements.txt
