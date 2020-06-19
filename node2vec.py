from graph import *
import pickle
from gensim.models import Word2Vec
# g = load_edgelist_graph("p2p-Gnutella08.edgelist")
g = load_mat_graph("blogcatalog.mat", "network","group")
# g = load_cora_graph() 
print("node number", len(g.nodes()))
print("Walking...")
walks = g.build_walk_corpus(80, 40, 32, 1, 1)
# f = open("walks_node_25_25", "rb")
# walks = pickle.load(f)
# f = open("walks_node_25_25", "wb")
# pickle.dump(walks,f)

# walks = g.build_walk_corpus(80, 40, 1, 1, 1)
# f = open("walks_node_1_1", "rb")
# walks = pickle.load(f)
# print("Training...")
model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, hs=0, workers=32, iter=1)
model.wv.save_word2vec_format("cora_25_25_iter1_factor2_lable50.embedding")
print(model.wv)
# from collections import defaultdict
# from six import iteritems
# def sparse2graph(x):
#     G = defaultdict(lambda: set())
#     cx = x.tocoo()
#     for i,j,v in zip(cx.row, cx.col, cx.data):
#         G[i].add(j)
#     return {str(k): [str(x) for x in v] for k,v in iteritems(G)}
# from scipy.io import loadmat
# mat = loadmat("blogcatalog.mat")

# A = mat["network"]
# graph = sparse2graph(A)
# labels_matrix = mat["group"]
# # print(graph)
# print(labels_matrix)