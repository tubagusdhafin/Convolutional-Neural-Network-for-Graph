import pandas as pd
import pickle

def list_node_neighbors(data):
    x= []
    for i in range(len(data)):
        neighbors = data[i]['neighbors']
        x.append(neighbors)
    return {'nodes neighbors': x}

if __name__ == "__main__":

	print("Opening graph from pickle file")
	f = open("datasets/imdb_action_romance.graph")
	data = pickle.loads(f.read())
	f.close()
	print("The data has been successfully open")

	print("Preprocessing data to DataFrame")
	dictionary_graph = { i : list_node_neighbors(data['graph'][i])
							for i in range(1000)}

	x = pd.DataFrame(dictionary_graph)
	x = x.T
	x['labels'] = data['labels']

	print("Save data to csv file")
	pd.DataFrame.to_csv(x,'datasets/imdb_action_romance.csv', index = False)