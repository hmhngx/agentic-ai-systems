# Test PDFs for the OCR ingestion pipeline

This directory holds PDFs for the Day 6 pipeline. Use either the **generated**
set (recommended for CI and first-time runs) or your own documents.

## Option A — generate deterministic test PDFs (recommended)

From the module root (`03-rag/3.4-OCR/`):

```powershell
python scripts/generate_test_pdfs.py
```

This writes three born-digital PDFs:

| File | Content |
| --- | --- |
| `test_academic.pdf` | ML topics, gradient descent, accuracy table |
| `test_report.pdf` | Quarterly revenue, market analysis, financial table |
| `test_tables.pdf` | Product specs, comparison table, benchmarks |

Each PDF has headings, prose, and captioned tables. Verification queries are
**derived automatically** from whatever is in Qdrant (`VERIFICATION_ADAPTIVE=true`,
the default), so these PDFs work without editing `src/verifier.py`.

For the original fixed query strings tuned to these three PDFs only, set
`VERIFICATION_ADAPTIVE=false` in `.env`.

## Option B — bring your own PDFs

Any born-digital or scanned PDF works if you:

1. Pass `--reingest` when replacing the collection (or use a fresh `COLLECTION_NAME`).
2. Use `--enable-ocr` for scanned pages (slow on CPU).
3. Keep `VERIFICATION_ADAPTIVE=true` so smoke-test queries match your tables and headings.

Optional override for the “find the table about …” query only:

```env
VERIFICATION_FIND_TABLE_QUERY=find the table about quarterly revenue
```

(Only used when `VERIFICATION_ADAPTIVE=false`.)

## Run against this directory

```powershell
python doc_ingest_pipeline.py --pdf-dir sample_pdfs/ --reingest
```
