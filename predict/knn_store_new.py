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
    keys = np.load(key_file, allow_pickle=True)
    values = np.load(value_file, allow_pickle=True)
    assert keys.shape[0] == values.shape[0]
    return keys, values

class KNN_STORE(object):
    def __init__(self, vector_dir, counter_dict=None, min_cnt=0, side="item"):
        self.min_cnt = min_cnt
        self.counter_dict = counter_dict
        self.side = side
        key_file = os.path.join(vector_dir, "{}_key.npy".format(side))
        value_file = os.path.join(vector_dir, "{}_value.npy".format(side))
        print("key_file: {}, value_file: {}".format(key_file, value_file))
        self.keys, self.values = load_np_ds(key_file, value_file)
        assert self.values.shape[0] == self.keys.shape[0]
        assert self.values.shape[1] == 1
        assert self.keys.shape[1] > 1
        self.keys, self.values = self.get_valid_ids(side="item")
        print("{} number: {}, {}_vector size: {}".format(side, self.keys.shape[0], side, self.keys.shape[1]))
        self.embedding_dim = self.keys.shape[1]
        self.index = self.build_index()


    def get_valid_ids(self, side="item"):
        count_array = np.array([self.counter_dict[id_] for id_ in self.values.reshape(-1).tolist()])
        valid_ids_idices = count_array > self.min_cnt
        return self.keys[valid_ids_idices], self.values[valid_ids_idices]


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
            # if embeddings.shape[0] < dstore_size_max:
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            # else:
            #     print("use IndexIVFPQ index")
            #     index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
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


    def build_index(self):
        return self._build_index(self.keys)

    def _search_top_n(self, query, index, topn=50):
        D, I = index.search(query, topn)
        return D, I
    def keys_search_topn(self, query, topn=50):
        D, I = self._search_top_n(query, self.index, topn=topn)
        values = self.values[I]
        return D, values

    def export_similar_values(self, output_file, topn=2):
        chunk_size = 10000
        chunk_num = self.keys.shape[0] // chunk_size
        with open(output_file, "w") as f:
            for chunk_idx in tqdm(range(chunk_num + 1)):
                user_keys_part = self.keys[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                values_part = self.values[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                D, I = self.keys_search_topn(query=user_keys_part, topn=topn)

                values = values_part.astype(str).reshape(-1).tolist()
                similar_values = I.reshape(len(values), -1).astype(str).tolist()
                assert I.shape[0] == len(values)
                for value, v2v_list, v2v_scores in tqdm(zip(values, similar_values, D.reshape(len(values), -1).tolist())):
                    v2v_list = [x.split("_")[0] for x in v2v_list] if self.side != "item" else [x + "00" for x in v2v_list]
                    value = value.split("_")[0] if self.side != "item" else value + "00"
                    v2v_results = ["{}:{:.3f}".format(x, y) for x, y in zip(v2v_list, v2v_scores)]
                    f.write("{}\t{}\n".format(value, ",".join(v2v_results)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_dir", default="", required=True, type=str)
    parser.add_argument("--side", default="item", required=True, type=str)
    parser.add_argument("--min_cnt", default=10, type=int)
    args = parser.parse_args()
    vector_dir = args.vector_dir
    output_file = "i2i.txt" if args.side == "item" else "u2u.txt"
    topn = 200 if args.side == "item" else 50
    with open(os.path.join(vector_dir, "user_item_counter.json"), "r") as f:
        user_item_counter_dict = json.load(f)
    counter_key = "user_ids_counter" if args.side == "user" else "item_ids_counter"
    knn_score = KNN_STORE(vector_dir=vector_dir, min_cnt=args.min_cnt, counter_dict=user_item_counter_dict[counter_key],
                          side=args.side)
    knn_score.export_similar_values(output_file=os.path.join(vector_dir, output_file), topn=topn)



