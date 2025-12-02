"""
BrandonBot Test Suite

Pytest-based tests for:
- Prequalifier (PQ) pipeline
- Output Validator (OV) safeguards:
  - Ethics (ME2-BERT)
  - Intent/Response (MS-MARCO)
  - PII (DeBERTa)
  - Confidence (BERT-tiny)
  - FEC Compliance (RAG + patterns)
  - Citations (Anchor resolution)
"""
