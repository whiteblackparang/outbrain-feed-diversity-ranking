import numpy as np


def build_similarity_maps(documents_categories, documents_topics):
    category_map = documents_categories.groupby("document_id")["category_id"].apply(set).to_dict()
    topic_map = documents_topics.groupby("document_id")["topic_id"].apply(set).to_dict()
    return category_map, topic_map


def jaccard_similarity(doc_a, doc_b, category_map, topic_map):
    cat_a, cat_b = category_map.get(doc_a, set()), category_map.get(doc_b, set())
    topic_a, topic_b = topic_map.get(doc_a, set()), topic_map.get(doc_b, set())

    union = cat_a | cat_b | topic_a | topic_b
    if not union:
        return 0
    intersection = (cat_a & cat_b) | (topic_a & topic_b)
    return len(intersection) / len(union)


def mmr_rerank(candidates, category_map, topic_map, lam=0.7, top_k=5):
    selected = []
    pool = candidates.copy()
    while pool and len(selected) < top_k:
        best, best_score = None, -np.inf
        for doc_id, relevance in pool:
            if not selected:
                sim = 0
            else:
                sim = max(
                    jaccard_similarity(doc_id, s, category_map, topic_map)
                    for s, _ in selected
                )
            mmr_score = lam * relevance - (1 - lam) * sim
            if mmr_score > best_score:
                best, best_score = (doc_id, relevance), mmr_score
        selected.append(best)
        pool.remove(best)
    return selected


def intra_list_diversity(selected_ids, category_map, topic_map):
    if len(selected_ids) < 2:
        return 0
    dists = []
    for i in range(len(selected_ids)):
        for j in range(i + 1, len(selected_ids)):
            sim = jaccard_similarity(selected_ids[i], selected_ids[j], category_map, topic_map)
            dists.append(1 - sim)
    return np.mean(dists)