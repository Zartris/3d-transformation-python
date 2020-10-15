import sys

from Transformations.transform import Transform


class Node(object):
    def __init__(self, name: str, leaf_node=True):
        self.name = name
        self.neighbors = {}
        self.edges = {}
        self.nodes_connected_to_me = set()
        self.nodes_reachable_from_me = set()
        self.leaf_node = leaf_node

    def __str__(self):
        return "Node: " + str(self.name)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self)

    def key_name(self):
        return self.name

    property(key_name)

    def add_edge(self, node, transform: Transform):
        """
        This is expensive!
        If you are just updating the node, call update instead
        :param node: target node
        :param transform: transform between the nodes
        :return:
        """
        if not isinstance(node, Node):
            print("The given node is not an instance of a Node")
            return False

        if node in self.edges:
            print("The given node already contains this edge. An update is then performed")
            self.update_edge(node, transform)
            return False
        self.leaf_node = False
        node.nodes_connected_to_me.add(self)
        return True

    def update_edge(self, node, transform: Transform) -> bool:
        """
        :param node:
        :param transform:
        :return:
        """
        if not isinstance(node, Node):
            print("The given node is not an instance of a Node")
            return False
        if node not in self.edges:
            print("The given node is not already recorded, please add the node before")
            return False
        self.edges[node] = transform
        return True

    def add_reachable_node(self, node):
        if node != self:
            self.nodes_reachable_from_me.add(node)


class TransformationManager:
    def __init__(self):
        self.edges = {}
        self.nodes = {}

        # Dirty edges is transformation already computed but is not updated.
        # So when we update an edge we clear this
        self.dirty_edges = {}

    def add_edge(self, from_node_str, to_node_str, transform):
        """
        Does not work with cycles
        :param from_node_str:
        :param to_node_str:
        :param transform:
        :return:
        """
        # Make sure each node exist in the graph
        from_node = self.get_or_create_node(from_node_str, leaf_node=False)
        to_node = self.get_or_create_node(to_node_str)

        # Create edge
        if from_node.add_edge(to_node, transform):
            # If new edge is added, we are updating all (Expensive)
            reachables_nodes = to_node.nodes_reachable_from_me.copy()
            reachables_nodes.add(to_node)
            self.update_nodes(from_node, reachables_nodes)

    def update_edge(self, from_node_str, to_node_str, transform):
        # Make sure each node exist in the graph
        from_node = self.get_or_create_node(from_node_str)
        to_node = self.get_or_create_node(to_node_str)

        # Update edge
        from_node.update_edge(to_node, transform)

    def getTransform(self, from_node_str, to_node_str):
        from_node = self.get_node_from_str(from_node_str)
        to_node = self.get_node_from_str(to_node_str)

        if not to_node in from_node.nodes_reachable_from_me:
            print(to_node, "is not reachable from", from_node)
            sys.exit()

        # Check for easy transform
        node = from_node
        transform = Transform()
        seen = []
        while True:
            # If we are neighbors to the end node:
            if to_node in node.edges.keys():
                transform.before(node.edges[to_node])
                break

            for neighborNode in node.edges.keys():
                if neighborNode not in seen and to_node in neighborNode.nodes_reachable_from_me:
                    # TODO: Loops in graph will be able to make this last for ever.
                    pass

        # TODO:: Would love to create this, but for now it is not needed for my project
        key = str(from_node.key_name) + "_" + str(to_node.key_name)
        if key in self.edges:
            return self.edges[key].transform
        if key in self.dirty_edges:
            return self.dirty_edges[key].transform

    def get_or_create_node(self, node_str: str, leaf_node=True) -> Node:
        if node_str not in self.nodes:
            self.nodes[node_str] = Node(node_str, leaf_node=leaf_node)
        return self.nodes[node_str]

    def get_node_from_str(self, node_str: str) -> Node:
        if node_str not in self.nodes:
            print(node_str, "is not part of the graph")
            sys.exit()
        return self.nodes[node_str]

    def update_nodes(self, current_node: Node, new_nodes):
        is_updated = False
        for node in new_nodes:
            if node != current_node and node not in current_node.nodes_reachable_from_me:
                current_node.add_reachable_node(node)
                is_updated = True
        if is_updated:
            for node in current_node.nodes_connected_to_me:
                self.update_nodes(node, new_nodes)


if __name__ == '__main__':
    TM = TransformationManager()
    TM.add_edge("E", "F", Transform())
    TM.add_edge("D", "E", Transform())
    TM.add_edge("D", "C", Transform())
    TM.add_edge("C", "B", Transform())
    TM.add_edge("B", "A", Transform())

    debug = 0
    TM.add_edge("A", "D", Transform())
    debug = 0
