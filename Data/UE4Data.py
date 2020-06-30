class Location:
    def __init__(self, location_dict):
        self.x = location_dict['x']
        self.y = location_dict['y']
        self.z = location_dict['z']


class Rotation:
    def __init__(self, rotation_dict):
        self.pitch = rotation_dict['pitch']
        self.roll = rotation_dict['roll']
        self.yaw = rotation_dict['yaw']


class State:
    def __init__(self, location, rotation):
        self.location = Location(location)
        self.rotation = Rotation(rotation)

    def __str__(self):
        return str(self.location) + "\n" + str(self.rotation)


# Wrapper
class Drone(State):
    def __init__(self, location, rotation, world_to_local, local_to_world):
        super().__init__(location, rotation)
        self.world_to_local = world_to_local
        self.local_to_world = local_to_world


# Wrapper
class Ship(State):
    def __init__(self, location, rotation):
        super().__init__(location, rotation)


def convert_data(data: dict) -> (Drone, Ship):
    drone = None
    ship = None
    for v_type in data.keys():
        location = data[v_type][0]
        rotation = data[v_type][1]
        if v_type == 'Drone':
            world_to_local = data[v_type][2]
            local_to_world = data[v_type][3]
            drone = Drone(location, rotation, world_to_local, local_to_world)
        elif v_type == 'Ship':
            ship = Ship(location, rotation)
    return drone, ship
