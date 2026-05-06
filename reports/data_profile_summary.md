# Data Profile Summary - PubMed Sample

## Sample
- Dataset: ccdv/pubmed-summarization
- Config: document
- Split: train[:100]
- Rows loaded: 100

## Columns
- article
- abstract

## Text Length Profile
- Average article length: 16,662.23 characters
- Average abstract length: 1,282.33 characters
- Longest article: 56,876 characters
- Shortest article: 0 characters
- Approx dataframe memory usage: 1.71 MB

## Data Quality Observations
- One article appears to have 0 characters.
- Articles are much longer than abstracts.
- There is no explicit ID column in the loaded dataset.
- Biomedical text is technical and variable in length.

## Scaling Concerns at 10k Articles
- Full article processing will become expensive.
- Tokenization and embedding generation may consume significant compute.
- RAG chunking over full articles may create many chunks.
- LLM summarization for every article would be costly and carbon-intensive.
- Repeated queries without caching would waste energy.

## Green AI Implication
The dataset profile supports the need for:
- Parquet storage
- caching
- lightweight models
- selective summarization
- Green Router decision logic
- avoiding unnecessary full-text LLM processing

## Worksheet B: Pipeline Schemas

| Layer | Columns expected | Validation rule |
| --- | --- | --- |
| Bronze articles | id, article, abstract, source_split, ingestion_ts | No empty id; article is not empty. |
| Silver articles | id, article_clean, abstract_clean, article_chars, abstract_chars, approx_tokens | Clean text exists; token count > 0. |
| Gold chunks | chunk_id, id, chunk_text, chunk_index, approx_tokens | No chunk exceeds the selected token budget. |
| Gold summaries | id, model, prompt_version, generated_summary, latency_s, input_tokens, output_tokens | Every generated summary links to one article and one prompt version. |
| Benchmark | experiment, variant, duration_s, energy_kwh, co2_kg, quality_score, notes, timestamp | Every benchmark row has a variant name and timestamp. |

Note: `approx_tokens` is estimated using `tiktoken` with the `cl100k_base` encoder. It is used as an approximate cost and processing-size indicator, not an exact model-specific token count.
