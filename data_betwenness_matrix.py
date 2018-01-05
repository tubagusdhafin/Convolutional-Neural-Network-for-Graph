import pandas as pd
import numpy as np
import networkx as nx
from sklearn.externals import joblib
import pickle

def list_node_neighbors(data):
    x= []
    for i in range(len(data)):
        neighbors = data[i]['neighbors']
        x.append(neighbors)
    return {'nodes neighbors': x}

def make_graph(data):
    g = nx.Graph()
    length_data = len(data)
    for i in range(length_data):
        number_neigbors = len(data[i])
        for j in range(number_neigbors):
            g.add_edge(str(i), str(j))
    return g

def make_betwenness_matrix(graph, nb_nodes):
    base_betweeness = nx.edge_betweenness(graph)
    item_betwenness = base_betweeness.items()
    matrix_betwenness = np.zeros((nb_nodes, nb_nodes))
    for i in range(len(item_betwenness)):
        j = int(item_betwenness[i][0][0])
        k = int(item_betwenness[i][0][1])
        matrix_betwenness[j][k] += item_betwenness[i][1]
    return matrix_betwenness


if __name__ == "__main__":

	f = open("datasets/imdb_action_romance.graph")
	data = pickle.loads(f.read())
	f.close()

	dictionary_graph = { i : list_node_neighbors(data['graph'][i])for i in range(1000)}
	x = pd.DataFrame(dictionary_graph)
	x = x.T
	x['labels'] = data['labels']

	data_betwenness_matrix = []
	data_graph_networkx = []
	print("Starting to convert graph data")
	for i in range(1000):
	    nb_nodes = len(x['nodes neighbors'][i])
	    graph = make_graph(x['nodes neighbors'][i])
	    betwenness_matrix = make_betwenness_matrix(graph, nb_nodes)
	    data_betwenness_matrix.append(betwenness_matrix)
	    data_graph_networkx.append(graph)
	    if i%1000 == 0:
	    	print i,(" Graph already convert to matrix and networkx")

	print("Save matrix to pickle file")
	joblib.dump(data_betwenness_matrix, 'datasets/data_betwenness_matrix.pkl')
	joblib.dump(data_graph_networkx, 'datasets/data_graph_networkx.pkl')
