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
from kivy.properties import StringProperty
from kivy.clock import mainthread, Clock
from kivy.config import Config
from datetime import timedelta
'''
# Bluetooth and D-Bus imports (Not used currently)
import dbus
import dbus.mainloop.glib
from gi.repository import GLib
import threading
'''

from shapely.geometry import LineString
from geopy.distance import geodesic

# Configure Kivy window size
Config.set('graphics', 'width', '600')   # Increased width
Config.set('graphics', 'height', '400')  # Increased height

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
    media_image = StringProperty("media_image.jpg")  # Initial media image

    # Edited code for ETA
    eta_info = StringProperty("ETA will be displayed here")
    # End Edits

    # Media Metadata Properties (Not used currently)
    title = StringProperty("Title: Unknown")
    artist = StringProperty("Artist: Unknown")
    album = StringProperty("Album: Unknown")

    def __init__(self, **kwargs):
        super(STHUDLayout, self).__init__(**kwargs)
        # Assign the global graph to an instance variable
        self.G = G
        if not os.path.exists(self.map_image_path):
            self.generate_default_map()

        # Start the Bluetooth media metadata listener (Not connected for now)
        #ble_thread = threading.Thread(target=self.ble_media_listener, daemon=True)
        #ble_thread.start()

        # Initialize variables for navigation simulation
        self.route_coords = []
        self.route_line = None
        self.route_nodes = []
        self.total_route_length = 0  # in meters
        self.current_distance = 0  # in meters
        self.speed_mps = 50 * 1000 / 3600  # 50 km/h in m/s (~13.8889 m/s)
        self.simulation_event = None

        # Initialize navigation instructions
        self.instructions = []
        self.current_instruction_index = 0
        self.distance_per_instruction = 0  # Will be calculated based on number of instructions

        # Initialize ETA update counter
        self.eta_update_counter = 0  # Counts seconds
        self.eta_update_interval = 1  # Start with 1 second interval

    def generate_default_map(self):
        # Create a default map image
        fig, ax = ox.plot_graph(
            self.G, 
            show=False, 
            close=True, 
            bgcolor='black', 
            node_size=0, 
            edge_color='gray'
        )
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

    # Edited code for ETA
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
        return timedelta(seconds=round(travel_time_seconds, 0))

    # End Edits

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
        route = self.plan_route(self.G, start_lat, start_lon, dest_address)
        if route:
            self.route_info = f"Route planned to {dest_address}."
            # Plot the route and update the map image
            self.plot_route(self.G, route)
            # Calculate and display ETA
            eta = self.calculate_eta(route)
            self.eta_info = f"ETA: {str(eta)}"
            # Generate turn-by-turn instructions
            self.instructions = self.generate_turn_by_turn_instructions(self.G, route)
            self.current_instruction_index = 0

            # Initialize navigation simulation before calculating distance_per_instruction
            self.initialize_navigation_simulation(route)

            # Calculate distance per instruction
            if self.instructions:
                self.distance_per_instruction = self.total_route_length / len(self.instructions) if len(self.instructions) > 0 else self.total_route_length
                print(f"Number of instructions: {len(self.instructions)}")
                print(f"Distance per instruction: {self.distance_per_instruction:.2f} meters")
            else:
                self.distance_per_instruction = self.total_route_length
                print("No instructions generated. Using total route length for distance per instruction.")

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

    def initialize_navigation_simulation(self, route):
        # Store the route nodes
        self.route_nodes = route  # list of node IDs

        # Extract route coordinates
        self.route_coords = [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in route]  # list of (lat, lon)

        # Create LineString
        self.route_line = LineString([(lon, lat) for lat, lon in self.route_coords])  # shapely expects (x, y) = (lon, lat)

        # Calculate total route length using geodesic distance
        self.total_route_length = 0
        for i in range(len(self.route_coords) - 1):
            start = self.route_coords[i]
            end = self.route_coords[i + 1]
            segment_distance = geodesic(start, end).meters
            self.total_route_length += segment_distance

        print(f"Total route length: {self.total_route_length:.2f} meters")
        # Initialize current distance
        self.current_distance = 0

        # Start simulation
        if self.simulation_event:
            Clock.unschedule(self.simulation_event)
        self.simulation_event = Clock.schedule_interval(self.update_simulation, 1)  # every second

    def update_simulation(self, dt):
        # Prevent division by zero
        if self.distance_per_instruction == 0:
            print("Error: distance_per_instruction is zero. Skipping instruction update.")
            return

        # Increment current distance
        self.current_distance += self.speed_mps * dt
        print(f"Current distance: {self.current_distance:.2f} meters")

        if self.current_distance >= self.total_route_length:
            self.current_distance = self.total_route_length
            if self.simulation_event:
                Clock.unschedule(self.simulation_event)
                self.simulation_event = None
            print("Reached destination")

        # Get current position as a point on the route
        current_position = self.get_position_at_distance(self.current_distance)

        # Update the map with current_position
        self.plot_simulated_map(current_position)

        # Update ETA based on remaining distance
        remaining_distance = self.total_route_length - self.current_distance
        eta_seconds = remaining_distance / self.speed_mps
        eta = timedelta(seconds=round(eta_seconds, 0))

        # Determine if within proximity threshold
        proximity_threshold = 500  # meters
        if remaining_distance <= proximity_threshold:
            # Within proximity threshold, update ETA every 5 seconds
            self.eta_update_counter += dt
            if self.eta_update_counter >= 5:
                self.eta_info = f"ETA: {str(eta)}"
                print(f"ETA Updated: {self.eta_info}")
                self.eta_update_counter = 0  # Reset counter
        else:
            # Outside proximity threshold, update ETA every second
            self.eta_info = f"ETA: {str(eta)}"
            print(f"ETA Updated: {self.eta_info}")
            self.eta_update_counter = 0  # Reset counter

        # Update navigation instructions based on distance
        if self.instructions:
            # Determine which instruction to show based on current distance
            instruction_index = int(self.current_distance // self.distance_per_instruction)
            if instruction_index < len(self.instructions):
                if instruction_index != self.current_instruction_index:
                    self.current_instruction_index = instruction_index
                    self.navigation_instruction = self.instructions[self.current_instruction_index]
                    print(f"Instruction Updated: {self.navigation_instruction}")
            else:
                # If all instructions have been displayed
                self.navigation_instruction = "You have arrived at your destination."

    def get_position_at_distance(self, distance):
        """
        Given a distance in meters, return the (lat, lon) position along the route.
        """
        distance_traversed = 0
        for i in range(len(self.route_coords) - 1):
            start = self.route_coords[i]
            end = self.route_coords[i + 1]
            segment_distance = geodesic(start, end).meters
            if distance_traversed + segment_distance >= distance:
                remaining_distance = distance - distance_traversed
                fraction = remaining_distance / segment_distance
                # Linear interpolation between start and end
                current_lat = start[0] + (end[0] - start[0]) * fraction
                current_lon = start[1] + (end[1] - start[1]) * fraction
                current_position = (current_lat, current_lon)
                return current_position
            distance_traversed += segment_distance
        # If distance exceeds total, return last point
        return self.route_coords[-1]

    def plot_simulated_map(self, current_position):
        """
        Plot the route and the current simulated position on the map.
        """
        # Plot the route
        fig, ax = ox.plot_graph_route(
            self.G,
            self.route_nodes,
            show=False,
            close=True,
            bgcolor='black',
            node_size=0,
            edge_color='gray',
            route_color='red',
            route_linewidth=3,
            bbox=None  # Let OSMNX decide the bounding box
        )

        # Plot the current position as a blue dot
        ax.plot(current_position[1], current_position[0], marker='o', markersize=10, markeredgecolor='blue', markerfacecolor='blue')

        # Save the map image to a fixed file to prevent accumulation
        image_path = "route_map_simulation.png"
        # Overwrite the existing simulation map
        fig.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Update the map_image_path to trigger Kivy to update
        self.map_image_path = image_path
    '''
    def ble_media_listener(self):
        print("Starting Bluetooth media listener...")
        # Initialize the D-Bus main loop
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()

        def properties_changed(interface, changed, invalidated, path=None):
            if interface != "org.bluez.MediaPlayer1":
                # Ignore signals from other interfaces
                return

            # Extract metadata
            title = "Unknown"
            artist = "Unknown"
            album = "Unknown"

            if 'Title' in changed or 'Artist' in changed or 'Album' in changed:
                title = changed.get("Title", "Unknown")
                artist = changed.get("Artist", "Unknown")
                album = changed.get("Album", "Unknown")
            elif 'Track' in changed and isinstance(changed['Track'], dbus.Dictionary):
                track_info = changed['Track']
                title = track_info.get('Title', "Unknown")
                artist = track_info.get('Artist', "Unknown")
                album = track_info.get('Album', "Unknown")

            # Debugging Output
            if title != "Unknown" or artist != "Unknown" or album != "Unknown":
                print(f"Now Playing: {title} by {artist} from {album} (Path: {path})")
            else:
                print(f"No metadata available for the current change (Path: {path})")

            # Schedule UI update
            Clock.schedule_once(lambda dt: self.update_media_info(title, artist, album))
        
        # Add a signal receiver for PropertiesChanged
        bus.add_signal_receiver(
            properties_changed,
            dbus_interface="org.freedesktop.DBus.Properties",
            signal_name="PropertiesChanged",
            path_keyword="path"
        )
        
        # Function to monitor for MediaPlayer1 interfaces
        def monitor_media_players():
            manager = dbus.Interface(bus.get_object("org.bluez", "/"), "org.freedesktop.DBus.ObjectManager")
            objects = manager.GetManagedObjects()
            for path, interfaces in objects.items():
                if "org.bluez.MediaPlayer1" in interfaces:
                    print(f"Connected to Media Player at {path}")
                    # Assign media_player for playback controls if needed
                    self.media_player = bus.get("org.bluez", path)

        # Initial scan for MediaPlayer1 interfaces
        #monitor_media_players()

        # Add a signal receiver for InterfacesAdded to detect new MediaPlayer1 interfaces
        def interfaces_added(path, interfaces):
            if "org.bluez.MediaPlayer1" in interfaces:
                print(f"New Media Player added at {path}")
                # Assign media_player for playback controls if needed
                self.media_player = bus.get("org.bluez", path)
        bus.add_signal_receiver(
            interfaces_added,
            dbus_interface="org.freedesktop.DBus.ObjectManager",
            signal_name="InterfacesAdded",
            path_keyword="path"
        )

        # Start the GLib main loop to listen for signals
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            loop.quit()
    '''

    @mainthread
    def update_media_info(self, title, artist, album):
        # Update the StringProperties
        if title != "Unknown":
            self.title = f"Title: {title}"
        if artist != "Unknown":
            self.artist = f"Artist: {artist}"
        if album != "Unknown":
            self.album = f"Album: {album}"

    def generate_turn_by_turn_instructions(self, G, route):
        """
        Generate turn-by-turn navigation instructions based on the route.

        Parameters:
            G (networkx.MultiDiGraph): The road network graph.
            route (list): List of node IDs representing the route.

        Returns:
            list: Navigation instructions.
        """
        instructions = []
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            edge_data = G.get_edge_data(current_node, next_node)
            if not edge_data:
                continue  # Skip if no edge data
            # Assuming the first edge in case of multiple edges
            edge = edge_data[list(edge_data.keys())[0]]
            street_name = edge.get('name', 'Unnamed Road')
            instructions.append(f"Proceed to {street_name}.")

        # If no instructions were generated, add a default instruction
        if not instructions:
            instructions.append("Proceed to your destination.")

        return instructions

    def plot_route(self, G, route):
        """
        Plot the planned route on the map.
        """
        import time
        timestamp = int(time.time())
        image_path = f"route_map_{timestamp}.png"

        # Get bounding box around the route for dynamic zoom
        lats = [G.nodes[node]['y'] for node in route]
        lons = [G.nodes[node]['x'] for node in route]
        north, south = max(lats), min(lats)
        east, west = max(lons), min(lons)
        margin = 0.01  # Add margin to the bounding box

        # Store route nodes for simulation
        self.route_nodes = route

        # Plot the route
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


class STHUDApp(App):
    def build(self):
        # Ensure default_map.png exists before building the UI
        layout = STHUDLayout()
        return layout


if __name__ == "__main__":
    STHUDApp().run()
