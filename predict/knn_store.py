import numpy as np
import faiss
import glob
from tqdm import tqdm
import argparse
from collections import Counter
import os
# import torch
import json
import math

# def align_loss(x, y, alpha=2):
#     return (x - y).norm(p=2, dim=1).pow(alpha).mean().numpy()
#
# def uniform_loss(x, t=2):
#     return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log().numpy()

def load_np_ds(key_file, value_file):
    item_keys = np.load(key_file, allow_pickle=True)
    item_values = np.load(value_file, allow_pickle=True)
    assert item_keys.shape[0] == item_values.shape[0]
    return item_keys, item_values

def nor(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

    # return x / np.sum(x)

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def getNDCG(rank_list, pos_items):
    pos_items_metrics = [x for x in pos_items if x in rank_list]
    relevance = np.ones_like(pos_items_metrics, dtype=float)
    it2rel = {it: r for it, r in zip(pos_items_metrics, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

class KNN_STORE(object):
    def __init__(self, vector_dir, user_item_counter_dict=None, item_min_cnt=1000, user_min_cnt=0):
        self.item_min_cnt = item_min_cnt
        self.user_min_cnt = user_min_cnt
        self.user_item_counter_dict = user_item_counter_dict
        user_key_file = os.path.join(vector_dir, "user_key.npy")
        user_value_file = os.path.join(vector_dir, "user_value.npy")
        item_key_file = os.path.join(vector_dir, "item_key.npy")
        item_value_file = os.path.join(vector_dir, "item_value.npy")

        self.item_keys, self.item_values = load_np_ds(item_key_file, item_value_file)
        self.user_keys, self.user_values = load_np_ds(user_key_file, user_value_file)
        # self.item_keys = self.item_keys.astype(np.float16)
        # print("user_keys MB: {}".format(self.user_keys.nbytes / 1024 / 1024))
        # self.user_keys = self.user_keys.astype(np.float16)
        # print("user_keys MB after fp16: {}".format(self.user_keys.nbytes / 1024 / 1024))
        assert self.item_values.shape[0] == self.item_keys.shape[0]
        assert self.user_values.shape[0] == self.user_keys.shape[0]
        assert self.item_values.shape[1] == 1
        assert self.item_keys.shape[1] > 1
        assert self.user_values.shape[1] == 1
        assert self.user_keys.shape[1] > 1
        self.item_keys, self.item_values = self.get_valid_ids(side="item")
        self.user_keys, self.user_values = self.get_valid_ids(side="user")
        print("item number: {}, item_vector size: {}".format(self.item_keys.shape[0], self.item_keys.shape[1]))
        print("user number: {}, user_vector size: {}".format(self.user_keys.shape[0], self.user_keys.shape[1]))

        assert self.item_keys.shape[1] == self.user_keys.shape[1]
        self.embedding_dim = self.item_keys.shape[1]

        # mixed embeddings
        self.user_num = self.user_keys.shape[0]
        self.item_num = self.item_keys.shape[0]
        # self.user_item_keys = np.concatenate([self.user_keys, self.item_keys], axis=0)
        # self.user_item_values = np.concatenate([self.user_values, self.item_values], axis=0)

        # build index
        self.item_index = self.build_item_index()
        self.user_index = self.build_user_index()
        # self.user_item_index = self.build_user_item_index()


    def get_valid_ids(self, side="item"):
        counter_key = "item_ids_counter" if side == "item" else "user_ids_counter"
        counter_value = self.item_min_cnt if side == "item" else self.user_min_cnt
        values = self.item_values if side == "item" else self.user_values
        keys = self.item_keys if side == "item" else self.user_keys
        count_array = np.array([self.user_item_counter_dict[counter_key][id_] for id_ in values.reshape(-1).tolist()])
        valid_ids_idices = count_array > counter_value
        return keys[valid_ids_idices], values[valid_ids_idices]


    def _build_index(self, embeddings):
        nlist = 100
        nprobe = nlist if embeddings.shape[0] < 1000000 else 10
        print("faiss nlist: {}, nprobe: {}".format(nlist, nprobe))
        dstore_size, embedding_dim = embeddings.shape[0], embeddings.shape[1]
        dstore_size_max = 10000000
        try:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            if embeddings.shape[0] < dstore_size_max:
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            else:
                print("use IndexIVFPQ index")
                index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
            assert not index.is_trained
            index = faiss.index_cpu_to_gpu(res, 0, index, co)
        except:
            print("use gpu failed.")
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            assert not index.is_trained
        if dstore_size > dstore_size_max:
            print("keep first 10000000 examples")
            index.train(embeddings[:dstore_size_max])
        else:
            index.train(embeddings)
        print("index trained.")
        index.add(embeddings)
        index.nprobe = nprobe
        return index


    def build_item_index(self):
        return self._build_index(self.item_keys)

    def build_user_index(self):
        return self._build_index(self.user_keys)

    def build_user_item_index(self):
        return self._build_index(self.user_item_keys)

    def _search_top_n(self, query, index, topn=50):
        D, I = index.search(query, topn)
        return D, I
    def item_search_topn(self, query, topn=50):
        D, I = self._search_top_n(query, self.item_index, topn=topn)
        items = self.item_values[I]
        return D, items
    def user_search_topn(self, query, topn=50):
        D, I = self._search_top_n(query, self.user_index, topn=topn)
        users = self.user_values[I]
        return D, users

    def user_item_search_topn(self, query, topn=50):
        D, I = self._search_top_n(query, self.user_item_index, topn=topn)
        is_user = I < self.user_num
        user_items = self.user_item_values[I]
        return D, user_items, is_user

    def export_similar_items(self, output_file, topn=2):
        D, I = self.item_search_topn(query=self.item_keys, topn=topn)
        items = self.item_values.astype(str).reshape(-1).tolist()
        similar_items = I.reshape(len(items), -1).astype(str).tolist()
        assert I.shape[0] == len(items)
        with open(output_file, "w") as f:
            for item, i2i_list, i2i_scores in tqdm(zip(items, similar_items, D.reshape(len(items), -1).tolist())):
                i2i_results = ["{}:{:.3f}".format(x, y) for x, y in zip(i2i_list, i2i_scores)]
                f.write("{}\t{}\n".format(item, ",".join(i2i_results)))

    def export_similar_users(self, output_file, topn=2):
        chunk_size = 10000
        chunk_num = self.user_keys.shape[0] // chunk_size
        with open(output_file, "w") as f:
            for chunk_idx in tqdm(range(chunk_num + 1)):
                user_keys_part = self.user_keys[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                user_values_part = self.user_values[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                D, I = self.user_search_topn(query=user_keys_part, topn=topn)

                users = user_values_part.astype(str).reshape(-1).tolist()
                similar_users = I.reshape(len(users), -1).astype(str).tolist()
                assert I.shape[0] == len(users)
                for user, u2u_list, u2u_scores in tqdm(zip(users, similar_users, D.reshape(len(users), -1).tolist())):
                    u2u_list = [x.split("_")[0] for x in u2u_list]
                    user = user.split("_")[0]
                    u2u_results = ["{}:{:.3f}".format(x, y) for x, y in zip(u2u_list, u2u_scores)]
                    f.write("{}\t{}\n".format(user, ",".join(u2u_results)))

    # def compute_align_uniform_loss(self, user, item):
    #
    #     user = torch.tensor(user)
    #     item = torch.tensor(item)
    #     return align_loss(user, item), uniform_loss(user), uniform_loss(item)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", default="", required=True, type=str)
    parser.add_argument("--item_min_cnt", default=10, type=int)
    parser.add_argument("--user_min_cnt", default=2, type=int)
    args = parser.parse_args()
    vector_dir = args.vector_dir
    with open(os.path.join(vector_dir, "user_item_counter.json"), "r") as f:
        user_item_counter_dict = json.load(f)
    knn_score = KNN_STORE(vector_dir=vector_dir, user_item_counter_dict=user_item_counter_dict,
                          item_min_cnt=args.item_min_cnt, user_min_cnt=args.user_min_cnt)
    knn_score.export_similar_items(output_file=os.path.join(vector_dir, "i2i.txt"), topn=200)
    knn_score.export_similar_users(output_file=os.path.join(vector_dir, "u2u.txt"), topn=50)



