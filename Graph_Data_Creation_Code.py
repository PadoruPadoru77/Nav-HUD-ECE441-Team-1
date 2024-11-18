import osmnx as ox

# Define the place and retrieve the graph with strict filtering
place_name = "Chicago, Illinois, USA"
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)

# Add lengths to edges that have geometry information
G = ox.distance.add_edge_lengths(G)

# Save the graph to a GraphML file
ox.save_graphml(G, "chicago_drive_UPDATED.graphml")

print("Graph saved with length attributes as 'chicago_drive_UPDATED.graphml'")
