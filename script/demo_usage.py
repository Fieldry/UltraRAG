from ultrarag.api import initialize, ToolCall

initialize(["retriever"], server_root="servers")

query_list = [
    "老人认为得了糖尿病就完全不能吃水果了，这种说法对吗？文献中关于老年糖友吃水果的时机和种类有什么建议？",
]

retriever_init_param_dict = {
    "model_name_or_path": "Qwen/Qwen3-Embedding-0.6B",
    "corpus_path": "corpora/text.jsonl",
    "batch_size": 32,
    "backend_configs": {},
}

ToolCall.retriever.retriever_init(
    **retriever_init_param_dict
)

# Generate embeddings
ToolCall.retriever.retriever_embed(
    embedding_path="corpora/embedding.npy",
    overwrite=True,
)

# Build index
ToolCall.retriever.retriever_index(
    embedding_path="corpora/embedding.npy",
    overwrite=True,
)

# Search
result = ToolCall.retriever.retriever_search(
    query_list=query_list,
    top_k=5,
)

retrieve_passages = result['ret_psg']
print(f"Retrieved {len(retrieve_passages)} results")
for i, passage in enumerate(retrieve_passages):
    print(f"Result {i+1}:")
    print(passage)
    print("-"*20)