def load_cora(cites_path, content_path):
    f1 =open(cites_path,"r")
    f2 = open(content_path, "r")
    cites = f1.readlines()
    contents = f2.readlines()
    # print(f2, contents)
    f1.close()
    f2.close()
    labels = []
    graph_edges = []
    for content in contents:
        data = content.split()
        paper_id = data[0]
        label_vector = [int(data[i]) for i in range(1, len(data)-1)]
        label = data[-1]
        labels.append([paper_id, label, label_vector])
    # print(labels)
    for cite in cites:
        data = cite.split()
        start = data[0]
        end = data[1]
        graph_edges.append([start, end])
    return labels, graph_edges
if __name__ == "__main__":
    print(load_cora("./cora.cites", "./cora.content"))
