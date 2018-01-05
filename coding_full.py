import pandas as pd
import numpy as np
import networkx as nx
from sklearn.externals import joblib
import pickle
from random import choice
from time import time 

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
        for j in data[i]:
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

def df_betweenness_cent(graph):   
    betweenness_cent = pd.DataFrame(nx.betweenness_centrality(graph), index =range(1))
    betweenness_cent =betweenness_cent.T
    betweenness_cent.columns = ['betweenness_centrality']
    return betweenness_cent

def df_degree_cent(graph):
    degree_cent = pd.DataFrame(nx.degree_centrality(graph), index =range(1))
    degree_cent = degree_cent.T
    degree_cent.columns = ['degree_centrality']
    return degree_cent

def df_eigen_cent(graph):
    eigen_cent = pd.DataFrame(nx.eigenvector_centrality(graph), index =range(1))
    eigen_cent = eigen_cent.T
    eigen_cent.columns = ['eigen_centrality']
    return eigen_cent

def df_close_cent(graph):
    close_cent = pd.DataFrame(nx.closeness_centrality(graph), index =range(1))
    close_cent = close_cent.T
    close_cent.columns = ['close_centrality']
    return close_cent

def df_load_cent(graph):
    load_cent = pd.DataFrame(nx.load_centrality(graph), index =range(1))
    load_cent = load_cent.T
    load_cent.columns = ['load_centrality']
    return load_cent

def df_centrality(graph,column_centrality):
    centrality = df_betweenness_cent(graph)
    for i in [df_close_cent, df_eigen_cent, df_degree_cent, df_load_cent]:
        cent1 = i(graph)
        centrality = centrality.join(cent1)
    centrality['node_number'] = centrality.index
    return centrality.sort_values(column_centrality, ascending = False)

def select_k_nodes(graph, centrality_type, k):
    dataframe_centrality = df_centrality(graph, centrality_type)
    k_nodes = []
    for i in range(k):
        k_nodes.append(int(dataframe_centrality['node_number'][i]))
    return k_nodes

def max_degree_neighbors(start_nodes, node_neighbors, node_not_select, index_graph):
    max_value = 0
    node_max = 0
    neighbors_node = [i for i in node_neighbors]
    neighbors = []
    for i in node_not_select:
        nb_start_nodes_neighbors = len(x['nodes neighbors'][index_graph][i])
        nb_neighbors = 0
        for j in range(nb_start_nodes_neighbors):
            if x['nodes neighbors'][index_graph][i][j] in neighbors_node:
                nb_neighbors +=1
        if nb_neighbors > max_value:
            max_value = nb_neighbors
            node_max = i
        neighbors.append(nb_neighbors)
    all_maxes = []
    for i in range(len(node_not_select)):
        if neighbors[i] == max_value:
            all_maxes.append(node_not_select[i])
    node_choice  = choice(all_maxes)
    return node_choice

def graph_neighbor(graph, start_nodes, nb_neighbors, index_graph, k=10):
    nb_start_nodes_neighbors = len(x['nodes neighbors'][index_graph][start_nodes])
    nb_graph_nodes = len(x['nodes neighbors'][index_graph])
    node_neighbors = [start_nodes]
    node_not_select = [i for i in range(nb_graph_nodes)]
    node_not_select.remove(start_nodes)
    if nb_start_nodes_neighbors<k:
        for i in range(nb_start_nodes_neighbors):
            node_neighbors.append(x['nodes neighbors'][index_graph][start_nodes][i])
            node_not_select.remove(x['nodes neighbors'][index_graph][start_nodes][i])
        while len(node_neighbors) <k:
            best_node = max_degree_neighbors(start_nodes,node_neighbors, node_not_select, index_graph)
            node_neighbors.append(best_node)
            node_not_select.remove(best_node)
    else:
        for i in x['nodes neighbors'][index_graph][start_nodes]:
            node_not_select.remove(i)
        for j in node_not_select:
            graph.remove_node(str(j))
        node_neighbors = select_k_nodes(graph, 'eigen_centrality', 10)
    return node_neighbors

if __name__ == "__main__":

    print("Starting to open graph data")
    t0 =time()
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
        if i%100 == 0:
            print i,(" Graph already convert to betweeness matrix and networkx")

    print("Save data graph and matrix to pickle file")
    joblib.dump(data_betwenness_matrix, 'datasets/data_betwenness_matrix.pkl')
    joblib.dump(data_graph_networkx, 'datasets/data_graph_networkx.pkl')

    print("Starting to select of 10 nodes for each graph in dataframe")
    k_nodes_dictionary = {i: select_k_nodes(data_graph_networkx[i], 'eigen_centrality', k=10) for i in range(len(data_graph_networkx))}
    dataframe_k_nodes = pd.DataFrame(k_nodes_dictionary)
    dataframe_k_nodes = dataframe_k_nodes.T
    dataframe_k_nodes.columns = ['node '+str(i) for i in range(1,11)]
    print("10 nodes for each graph has been selected in dataframe")

    print("Save dataframe to pickle file")
    joblib.dump(dataframe_k_nodes, 'datasets/dataframe_k_nodes.pkl')

    print("Select 9 neighbor nodes for each selected 10 nodes ")
    k_nodes_with_neighbors = dataframe_k_nodes
    for jdx in range(1,11):
        data_graph = joblib.load('datasets/data_graph_networkx.pkl')
        k_nodes_with_neighbors['neighbor of node '+str(jdx)] =[graph_neighbor(graph = data_graph[idx], 
                                                                       start_nodes=k_nodes_with_neighbors['node '+str(jdx)][idx],
                                                                       nb_neighbors=10, 
                                                                     index_graph=idx, k=10) for idx in range(1000)]

        print "Neighbor of node ", jdx, "has been selected"  

    print("Save dataframe to pickle file")
    joblib.dump(k_nodes_with_neighbors, 'datasets/k_nodes_with_neighbors.pkl')

    print("Starting to make betweeness matrix for each neighbor of 10 nodes for each graph")
    betwenness_neighbor_matrix = []
    for l in range(1000):
        matrix_betwenness_neighbor = []
        for i in range(1,11):
            betwenness_neighbor = np.zeros([10, 10])
            for j in range(10):
                for k in range(10):
                    node_j = dataframe_k_nodes['neighbor of node '+str(i)][l][j]
                    node_k = dataframe_k_nodes['neighbor of node '+str(i)][l][k]
                    betwenness_neighbor[j][k] = data_betwenness_matrix[l][node_j][node_k]
            matrix_betwenness_neighbor.append(betwenness_neighbor)
        betwenness_neighbor_matrix.append(matrix_betwenness_neighbor)
        if l%100 == 0:
            print i,("Graph already convert to 10 betweeness neighbor matrix and")

    print("Save matrix to pickle file")
    joblib.dump(betwenness_neighbor_matrix, 'datasets/betwenness_neighbor_matrix.pkl')
