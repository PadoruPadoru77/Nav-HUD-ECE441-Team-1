import osmnx as ox
import os
import networkx as nx
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty
from kivy.clock import mainthread
from kivy.config import Config
from datetime import timedelta

Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '380')

# Load the kv file explicitly
Builder.load_file("st_hud.kv")

# Load the saved map graph
G = ox.load_graphml("chicago_drive_UPDATED.graphml")


class STHUDLayout(FloatLayout):
    bluetooth_status = StringProperty("Not Connected")
    address = StringProperty("Enter Address")
    route_info = StringProperty("Route will be displayed here")
    navigation_instruction = StringProperty("Navigation instructions will appear here")
    map_image_path = StringProperty("default_map.png")  # Initial map image
    
    #Edited code for ETa- Michael
    #
    #
    #
    eta_info = StringProperty("ETA will be displayed here")
    #End Edits

    def __init__(self, **kwargs):
        super(STHUDLayout, self).__init__(**kwargs)
        # Generate an initial map image if needed
        self.G = G  # Assign the global graph to an instance variable
        if not os.path.exists(self.map_image_path):
            self.generate_default_map()

    def generate_default_map(self):
        # Create a default map image
        fig, ax = ox.plot_graph(G, show=False, close=True, bgcolor='black', node_size=0, edge_color='gray')
        fig.savefig(self.map_image_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    @mainthread
    def on_address_entered(self, address_text):
        self.address = address_text.strip()
        if not self.address:
            self.route_info = "Please enter a valid address."
            return
        self.route_info = f"Planning route to: {self.address}"
        # Start route planning
        self.plan_route_to_address(self.address)


    #Edited code for ETA - Michael
    #
    #
    #
    #
    #Calculate ETA
    def calculate_eta(self, route, avg_speed_kmh=50):
        route_length = 0
        for u, v in zip(route[:-1], route[1:]):
            # Check each multi-edge between nodes u and v
            length = None
            for key, data in self.G.get_edge_data(u, v).items():
                length = data.get('length')
                if length:
                    route_length += length
                    break  # Stop after finding the first valid length for multi-edges
            if length is None:
                print(f"Warning: Missing length for edge ({u}, {v}). Using default length of 50 meters.")
                route_length += 50  # Default length in meters if missing

        avg_speed_mps = avg_speed_kmh * 1000 / 3600
        travel_time_seconds = route_length / avg_speed_mps
        return timedelta(seconds=travel_time_seconds)

   #End Edits

    def plan_route_to_address(self, dest_address):
        # Use predefined location or GPS data
        use_gps = False  # Set to True to use GPS data
        if not use_gps:
            start_lat, start_lon = 41.878876, -87.635915  # Willis Tower
        else:
            start_lat, start_lon = self.get_current_location()
            if start_lat is None or start_lon is None:
                self.route_info = "GPS data not available."
                return

        # Plan the route
        route = self.plan_route(G, start_lat, start_lon, dest_address)
        if route:
            self.route_info = f"Route planned to {dest_address}."
            # Plot the route and update the map image
            self.plot_route(G, route)
            # Calculate and display ETA
            eta = self.calculate_eta(route)
            self.eta_info = f"ETA: {str(eta)}"
            # Generate turn-by-turn instructions
            instructions = self.generate_turn_by_turn_instructions(G, route)
            self.navigation_instruction = instructions
        else:
            self.route_info = f"Failed to plan the route to {dest_address}."
            self.navigation_instruction = "No navigation instructions available."
            # Optionally, reset the map image to default
            self.map_image_path = "default_map.png"

    def get_coordinates_from_address(self, address):
        geolocator = Nominatim(user_agent="st_hud")
        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"Address '{address}' not found.")
                return None, None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None, None

    def plan_route(self, G, start_lat, start_lon, dest_address):
        # Get destination coordinates from address
        dest_lat, dest_lon = self.get_coordinates_from_address(dest_address)
        if dest_lat is None or dest_lon is None:
            print("Invalid destination address.")
            return None

        # Get nearest nodes to the start and destination coordinates
        try:
            orig_node = ox.nearest_nodes(G, X=start_lon, Y=start_lat)
            dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)
        except Exception as e:
            print(f"Error finding nearest nodes: {e}")
            return None

        # Calculate the shortest path
        try:
            route = nx.shortest_path(G, orig_node, dest_node, weight='length')
            return route
        except Exception as e:
            print(f"Error calculating shortest path: {e}")
            return None

    def plot_route(self, G, route):
        import time
        timestamp = int(time.time())
        image_path = f"route_map_{timestamp}.png"

        # Get bounding box around the route for dynamic zoom
        lats = [G.nodes[node]['y'] for node in route]
        lons = [G.nodes[node]['x'] for node in route]
        north, south = max(lats), min(lats)
        east, west = max(lons), min(lons)
        margin = 0.01  # Add margin to the bounding box

        fig, ax = ox.plot_graph_route(
            G,
            route,
            show=False,
            close=True,
            bgcolor='black',
            node_size=0,
            edge_color='gray',
            route_color='red',
            route_linewidth=3,
            bbox=(north + margin, south - margin, east + margin, west - margin)
        )
        fig.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        self.map_image_path = image_path  # Update the image path with new filename


    def generate_turn_by_turn_instructions(self, G, route):
        # Simple placeholder implementation
        instructions = "Proceed along the highlighted route."
        # More advanced instruction generation can be implemented here
        return instructions

    def get_current_location(self):
        # Implement GPS data retrieval
        try:
            import gpsd
            gpsd.connect()
            packet = gpsd.get_current()
            if packet.mode >= 2:  # Mode 2D or 3D fix
                return packet.lat, packet.lon
            else:
                print("GPS has no fix.")
                return None, None
        except Exception as e:
            print(f"GPS error: {e}")
            return None, None

class STHUDApp(App):
    def build(self):
        # Ensure default_map.png exists before building the UI
        layout = STHUDLayout()
        return layout


if __name__ == "__main__":
    STHUDApp().run()
