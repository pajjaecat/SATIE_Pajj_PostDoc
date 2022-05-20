import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Lines.csv", header=1)
df2 = pd.read_csv("Loads.csv", header=2)

G = nx.Graph()

edges = []

buses_nodes = []
for index, row in df.iterrows() :
	bus_from = row["Bus1"]
	bus_to = row["Bus2"]
	buses_nodes.append(bus_from)
	buses_nodes.append(bus_to)
	line = row["Name"]
	G.add_node(bus_from)
	G.add_node(bus_to)
	G.add_edge(bus_from, bus_to)

load_nodes = []
for index, row in df2.iterrows() :
	bus = row["Bus"]
	load_nodes.append(row["Name"])
	G.add_node(row["Name"])
	G.add_edge(bus, row["Name"])
	
labels = {}    
for node in G.nodes():
    if node in load_nodes:
        labels[node] = node	
	
pos=nx.kamada_kawai_layout(G)

nx.draw_networkx_nodes(G, pos, nodelist=load_nodes, node_color="r", alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=buses_nodes, node_size=50, node_color="b", alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=[buses_nodes[0]], node_size=500, node_color="g", alpha=0.2)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=labels, verticalalignment="top", font_size=7)
plt.show()
