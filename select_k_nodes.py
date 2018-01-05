import pandas as pd
import numpy as np
import networkx as nx
from sklearn.externals import joblib
import pickle


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

if __name__ == "__main__":

	data_graph = joblib.load('datasets/data_graph_networkx.pkl')

	k_nodes_dictionary = {i: select_k_nodes(data_graph[i], 'eigen_centrality', k=10) for i in range(len(data_graph))}
	dataframe_k_nodes = pd.DataFrame(k_nodes_dictionary)
	dataframe_k_nodes = dataframe_k_nodes.T
	dataframe_k_nodes.columns = ['node '+str(i) for i in range(1,11)]
	dataframe_k_nodes.head()

	pd.DataFrame.to_csv(dataframe_k_nodes, 'datasets/selected_k_nodes.csv', index = False)