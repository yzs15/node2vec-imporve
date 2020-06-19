from collections import defaultdict, Iterable
import random
import threading
from multiprocessing import Process, Queue
from scipy.sparse.base import issparse
from scipy.io import loadmat
from citeseer.load_cora import load_cora
class Graph(defaultdict):
    def __init__(self, factor_func=None):
        super(Graph, self).__init__(defaultdict)
        self.label ={}
        if factor_func == None:
            self.factor_func = self.default_factor_func
        else:
            self.factor_func = factor_func
    def default_factor_func(self, label, start, candidate):
        factor = 1
        if label.__contains__(candidate) and label.__contains__(start):
            if label[candidate] == label[start]:
                factor = 2
        return factor
    def nodes(self):
        return list(self.keys())
    # def pre_calculate_prop(self, p=1, q=1):
    #     nodes = self.nodes()
    #     for node in nodes:
    #         candidates = list(self[node].keys())

    def random_walk(self, node,length, p=1, q=1):
        walk = [node]
        if len(list(self[node].keys())) <= 0:
            return walk
        start = node
        length = length-1
        p_inver = 1/p
        q_inver = 1/q
        while length>0:
            if len(walk) < 2:
                next_node = random.choice(list(self[start].keys()))
                walk.append(next_node)
                start = next_node
            else:
                t = walk[-2]
                candidates = list(self[start].keys())
                props = []
                total = 0
                for candidate in candidates:
                    factor = self.factor_func(self.label, start, candidate)
                    if candidate == t:
                        prop = p_inver * factor
                    elif self[t].__contains__(candidate):
                        prop = 1*factor
                    else:
                        prop = q_inver*factor
                    total += prop
                    props.append(prop)
                if len(props) <= 0:
                    return walk
                p_choice = random.random() * total
                index = 0
                while p_choice >= props[index]:
                    p_choice = p_choice - props[index]
                    index += 1
                    if index >= len(props):
                        index = len(props)
                        break
                next_node = candidates[index]
                walk.append(next_node)
                start = next_node
            length = length - 1
        return walk
    def build_walk_corpus_func(self, queue, num_paths, path_length, nodes = [], p=1, q=1 ):
        # print("build_walk_corpus_func", len(nodes))
        walks = []
        for i in range(num_paths):
            random.shuffle(nodes)
            print("build_walk_corpus_func", num_paths, i)
            for node in nodes:
                walks.append(self.random_walk(node, path_length, p, q))
            #     print(type(walks[0][0]))
            #     break
            # break
        queue.put(walks)
        # self.lock.acquire()
        # self.walks = self.walks + walks
        # self.lock.release()
        # return walks
    def build_walk_corpus(self, num_paths, path_length, process_num = 8, p=1, q=1):
        nodes = self.nodes()
        num_per_t = len(nodes) // process_num
        # self.lock = threading.Lock()
        walks = []
        l = []
        queues = [Queue() for i in range(process_num)]
        for i in range(process_num):
            if i == process_num-1:
                proc = Process(target=self.build_walk_corpus_func, args=(queues[i], num_paths, path_length, nodes[i*num_per_t:],p,q, ))
            else:
                proc = Process(target=self.build_walk_corpus_func, args=(queues[i], num_paths, path_length, nodes[i*num_per_t:(i+1)*num_per_t],p,q, ))
            proc.start()
            l.append(proc)
        for i in range(process_num):
            walks = walks + queues[i].get()
        for proc in l:
            proc.join()
        print(len(walks))
        return walks

def load_cora_graph():
    labels, edges = load_cora("citeseer/citeseer.cites","citeseer/citeseer.content")
    G = Graph()
    for edge in edges:
        G[str(edge[0])][str(edge[1])] = {}
    for label in labels:
        G.label[label[0]] = label[1]
    return G

def load_edgelist_graph(path):
    G = Graph()
    f = open(path)
    edges = f.readlines()
    for edge in edges:
        node = edge.split()
        if len(node) < 2:
            continue
        start = node[0]
        end = node[1]
        if start == end:
            continue
        # print(start, end)
        if G.__contains__(start):
            G[start][end] = []
        else:
            G[start] = {end:[]}

        if G.__contains__(end):
            G[end][start] = []
        else:
            G[end] = {start:[]}
    return G
def load_mat_graph(path, variable_name, label_matrix_name = None, undirected=True):
    mat_varables = loadmat(path)
    mat_matrix = mat_varables[variable_name]
    labels_matrix = None
    if mat_matrix != None:
        labels_matrix = mat_varables[label_matrix_name]
    return from_numpy(mat_matrix,labels_matrix, undirected)

def from_numpy(x, labels_matrix=None, undirected=True):
    G = Graph()
    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            if i==j:
                continue
            G[str(i)][str(j)] = {}
            if undirected:
                G[str(j)][str(i)] = {}
    else:
      raise Exception("Dense matrices not yet supported.")
    if labels_matrix != None:
        if issparse(labels_matrix):
            cx = labels_matrix.tocoo()
            for i,j,v in zip(cx.row, cx.col, cx.data):
                if random.random() > 0.5:
                    G.label[str(i)] = j
            # print(len(cx.row), len(cx.col))
            # exit()
        else:
          raise Exception("Dense matrices not yet supported.")

    # if undirected:
    #     G.make_undirected()

    # G.make_consistent()
    return G

if __name__ == "__main__":
    # G = load_graph("p2p-Gnutella08.edgelist")
    # print((list(G["6296"].keys())))
    # print(G.random_walk("6296", 10, 1))
    # print(G.nodes())
    load_cora_graph()