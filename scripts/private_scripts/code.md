# Convert MCQ to BioASQ format (for retrieval):

python scripts/convert_orange_to_bioasq.py \
    --input data/train_test_dataset/orange_qa_MCQ_test.jsonl \
    --output data/train_test_dataset_bioasq_style/orange_qa_MCQ_test_bioasq.json
python scripts/convert_orange_to_bioasq.py \
    --input data/train_test_dataset/orange_qa_MCQ-con_test.jsonl \
    --output data/train_test_dataset_bioasq_style/orange_qa_MCQ-con_test_bioasq.json

# build index
python scripts/shared_scripts/index/build_bm25_index_from_jsonl_shards.py \
  --jsonl_glob "/Users/yun/develop/RAG-orange/data/orange_docs_chunks.jsonl" \
  --index_path "/Users/yun/develop/RAG-orange/index/bm25" \
  --threads 4 \
  --overwrite

python scripts/shared_scripts/index/build_dense_hnsw_index_from_jsonl_shards.py \
  --jsonl_glob "data/orange_docs_chunks.jsonl" \
  --out_dir "index/bg3_m3" \
  --model_name "BAAI/bge-m3" \
  --device "mps" \
  --batch_size 8 \
  --dedup_pmids 

scripts/shared_scripts/run_retrieval_rerank_pipeline.sh --config scripts/private_scripts/config.env