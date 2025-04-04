import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        self.neighbors.append(node)
        node.neighbors.append(self)

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes_dict[(r, c)] = Node((r, c))

    for (r, c), node in nodes_dict.items():
        if (r-1, c) in nodes_dict:
            node.neighbors.append(nodes_dict[(r-1, c)])
        if (r+1, c) in nodes_dict:
            node.neighbors.append(nodes_dict[(r+1, c)])
        if (r, c+1) in nodes_dict:
            node.neighbors.append(nodes_dict[(r, c+1)])
        if (r, c-1) in nodes_dict:
            node.neighbors.append(nodes_dict[(r, c-1)])


    start_node = nodes_dict.get((0,0), None)
    goal_node = nodes_dict.get((rows-1, cols-1), None)

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    visited = set() 
    queue = deque([start_node])
    parent_map = {start_node: None}  

    while queue:
        node = queue.popleft()  
        visited.add(node)
        
        if node == goal_node:
            path = []
            while node is not None:
                path.append((node.value[0], node.value[1])) 
                node = parent_map[node]
            return path[::-1]  

        for neighbor in sorted(node.neighbors, key=lambda x: x.value):
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                parent_map[neighbor] = node  
    


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    visited = set()
    stack = [start_node]
    parent_map = {start_node: None} 

    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node == goal_node:
            path = []
            while node is not None:
                path.append((node.value[0], node.value[1]))  
                node = parent_map[node]
            return path[::-1] 
        
        for neighbor in sorted(node.neighbors, key=lambda x: x.value, reverse=True):
            if neighbor not in visited:
                stack.append(neighbor)
                parent_map[neighbor] = node 

    


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: manhattan_distance(start_node, goal_node)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current.value)
                current = came_from[current]
            path.append(start_node.value)
            return path[::-1]
        
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1  
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal_node)
                if neighbor not in [n[1] for n in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None
def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    
    (r1, c1) = node_a.value
    (r2,c2) = node_b.value
    return np.abs(r1 - r2) + np.abs(c1 - c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    if start_node == goal_node:
        return [(start_node.row, start_node.col)]
    
    frontier_start = deque([start_node])
    frontier_goal = deque([goal_node])
    
    came_from_start = {start_node: None}
    came_from_goal = {goal_node: None}
    
    while frontier_start and frontier_goal:
        if frontier_start:
            current_start = frontier_start.popleft()
            for neighbor in current_start.neighbors:
                if neighbor not in came_from_start:
                    frontier_start.append(neighbor)
                    came_from_start[neighbor] = current_start
                if neighbor in came_from_goal:
                    path_from_start = reconstruct_path(current_start, came_from_start)
                    path_from_goal = reconstruct_path(neighbor, came_from_goal)
                    return path_from_start + path_from_goal[::-1]
        
        if frontier_goal:
            current_goal = frontier_goal.popleft()
            for neighbor in current_goal.neighbors:
                if neighbor not in came_from_goal:
                    frontier_goal.append(neighbor)
                    came_from_goal[neighbor] = current_goal
                if neighbor in came_from_start:
                    path_from_start = reconstruct_path(neighbor, came_from_start)
                    path_from_goal = reconstruct_path(current_goal, came_from_goal)
                    return path_from_start + path_from_goal[::-1]
    return None


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    current = start_node
    path = [current.value]
    cost = manhattan_distance(current, goal_node)
    
    while temperature > min_temperature:
        candidate = np.random.choice(current.neighbors)
        next_cost = manhattan_distance(candidate, goal_node)
        
        if next_cost < cost:
            current = candidate
            cost = next_cost
        else:
            probability = math.exp(-(next_cost - cost) / temperature)
            if np.random.rand() < probability:
                current = candidate
                cost = next_cost
        
        path.append(current.value)
        temperature *= cooling_rate
        
        if current == goal_node:
            return path
    
    return None




###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    path = []
    current_node = end_node
    while current_node is not None:
        path.append((current_node.value[0], current_node.value[1]))  
        current_node = parent_map.get(current_node, None)
    return path[::-1]


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
