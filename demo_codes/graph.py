from Stack_Queue_DoublyLinkedList import *
class UnionFind:
    def __init__(self, n):
        """Initialize the Union-Find structure with n elements."""
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n           # Rank is used for Union by Rank optimization

    def find(self, x):
        """Find the root of the element x with path compression."""
        if self.parent[x] != x:
            # Path compression: make the root of x's set as the direct parent of x
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y using Union by Rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by Rank: attach the smaller tree under the larger tree
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def connected(self, x, y):
        """Check if elements x and y are in the same set."""
        return self.find(x) == self.find(y)
# Define custom exceptions
class GraphException(Exception):
    """General exception for graphs"""
    pass

class VertexNotFoundException(GraphException):
    """Exception for vertex not found"""
    def __init__(self, vertex):
        self.message = f"Vertex '{vertex}' not found in the graph."
        super().__init__(self.message)

class EdgeNotFoundException(GraphException):
    """Exception for edge not found"""
    def __init__(self, u, v):
        self.message = f"Edge from '{u}' to '{v}' not found in the graph."
        super().__init__(self.message)

class InvalidGraphException(GraphException):
    """Exception for invalid graph structure"""
    def __init__(self, message="Invalid graph structure."):
        self.message = message
        super().__init__(self.message)

class NegativeWeightException(GraphException):
    """Exception for negative weight in unsupported graphs"""
    def __init__(self, u, v, weight):
        self.message = f"Negative weight '{weight}' detected on edge from '{u}' to '{v}'."
        super().__init__(self.message)


# Base class Graph (unweighted)
class Graph:
    def __init__(self, directed=False):
        """Initialize the graph with an option for directed or undirected"""
        self.__adjacency_list = {}  # Private attribute to store adjacency list
        self.__directed = directed  # Private attribute to store direction
        self.__num_vertices = 0  # Thêm biến đếm số đỉnh
        self.__num_edges = 0     # Thêm biến đếm số cạnh
    
    def __repr__(self):
        """Return the string representation of the graph"""
        return f"Graph(directed={self.__directed}, vertices={len(self.__adjacency_list)})"
    
    @property
    def num_vertices(self):
        """Trả về số đỉnh trong đồ thị"""
        return self.__num_vertices

    @property
    def num_edges(self):
        """Trả về số cạnh trong đồ thị"""
        return self.__num_edges

    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        if vertex in self.__adjacency_list:
            raise InvalidGraphException(f"Vertex '{vertex}' already exists.")
        self.__adjacency_list[vertex] = []
        self.__num_vertices += 1  # Cập nhật số đỉnh
    
    def add_edge(self, u, v):
        """Add an edge between two vertices"""
        if u not in self.__adjacency_list:
            raise VertexNotFoundException(u)
        if v not in self.__adjacency_list:
            raise VertexNotFoundException(v)
        
        self.__adjacency_list[u].append(v)
        self.__num_edges += 1  # Cập nhật số cạnh
        if not self.__directed:
            self.__adjacency_list[v].append(u)
            # Không tăng num_edges vì đây là cùng một cạnh
    
    def get_edges(self):
        """Return the list of all edges in the graph"""
        edges = []
        for u in self.__adjacency_list:
            for v in self.__adjacency_list[u]:
                edges.append((u, v))
        return edges

    def remove_edge(self, u, v):
        """Remove an edge between two vertices"""
        if u not in self.__adjacency_list:
            raise VertexNotFoundException(u)
        if v not in self.__adjacency_list[u]:
            raise EdgeNotFoundException(u, v)
        
        self.__adjacency_list[u].remove(v)
        self.__num_edges -= 1  # Cập nhật số cạnh
        if not self.__directed:
            self.__adjacency_list[v].remove(u)
            # Không giảm num_edges vì đây là cùng một cạnh

    def is_directed(self):
        """Public getter method to check if the graph is directed"""
        return self.__directed

    def get_adjacency_list(self):
        """Public getter method to return a copy of the adjacency list"""
        return {k: list(v) for k, v in self.__adjacency_list.items()}

    def bfs(self, start_vertex):
        """Breadth-First Search (BFS) using custom Queue"""
        visited = set()
        queue = Queue()
        queue.enqueue(start_vertex)
        traversal_order = []
        
        while not queue.is_empty():
            vertex = queue.dequeue()
            if vertex not in visited:
                visited.add(vertex)
                traversal_order.append(vertex)
                
                for neighbor in self.__adjacency_list[vertex]:
                    if neighbor not in visited:
                        queue.enqueue(neighbor)
        
        return traversal_order

    def dfs(self, start_vertex):
        """Depth-First Search (DFS) using custom Stack"""
        visited = set()
        stack = Stack()
        stack.push(start_vertex)
        traversal_order = []
        
        while not stack.is_empty():
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                traversal_order.append(vertex)
                
                for neighbor in self.__adjacency_list[vertex]:
                    if neighbor not in visited:
                        stack.push(neighbor)
        
        return traversal_order

    def beam_search(self, start_vertex, beam_width):
        """Beam Search using custom Queue"""
        visited = set([start_vertex])
        queue = Queue()
        queue.enqueue(start_vertex)
        beam_results = []

        while not queue.is_empty():
            current_level = []
            for _ in range(min(queue.size(), beam_width)):
                vertex = queue.dequeue()
                beam_results.append(vertex)
                current_level.extend([neighbor for neighbor in self.__adjacency_list[vertex] if neighbor not in visited])

            for node in current_level[:beam_width]:
                queue.enqueue(node)
                visited.add(node)

        return beam_results

    def density(self):
        """Tính mật độ của đồ thị"""
        if self.num_vertices <= 1:
            return 0
        
        if self.__directed:
            return self.num_edges / (self.num_vertices * (self.num_vertices - 1))
        else:
            return (2 * self.num_edges) / (self.num_vertices * (self.num_vertices - 1))

    def degree_centrality(self):
        """Tính độ trung tâm bậc (chuẩn hóa) cho mỗi đỉnh"""
        if self.num_vertices <= 1:
            return {v: 0 for v in self.__adjacency_list}
        
        centrality = {}
        for vertex in self.__adjacency_list:
            if self.__directed:
                in_degree = sum(1 for v in self.__adjacency_list if vertex in self.__adjacency_list[v])
                out_degree = len(self.__adjacency_list[vertex])
                centrality[vertex] = {
                    'in': in_degree / (self.num_vertices - 1),
                    'out': out_degree / (self.num_vertices - 1),
                    'total': (in_degree + out_degree) / (2 * (self.num_vertices - 1))
                }
            else:
                degree = len(self.__adjacency_list[vertex])
                centrality[vertex] = degree / (self.num_vertices - 1)
        
        return centrality

    def _shortest_paths_bfs(self, start_vertex):
        """Helper method để tìm đường đi ngắn nhất từ start_vertex đến tất cả các đỉnh khác"""
        distances = {v: float('inf') for v in self.__adjacency_list}
        distances[start_vertex] = 0
        predecessors = {v: [] for v in self.__adjacency_list}
        
        # Sử dụng BFS để tìm đường đi ngắn nhất
        visited = set()
        queue = Queue()
        queue.enqueue(start_vertex)
        
        while not queue.is_empty():
            vertex = queue.dequeue()
            if vertex not in visited:
                visited.add(vertex)
                
                for neighbor in self.__adjacency_list[vertex]:
                    if distances[neighbor] > distances[vertex] + 1:
                        distances[neighbor] = distances[vertex] + 1
                        predecessors[neighbor] = [vertex]
                        queue.enqueue(neighbor)
                    elif distances[neighbor] == distances[vertex] + 1:
                        predecessors[neighbor].append(vertex)
                        queue.enqueue(neighbor)
        
        return distances, predecessors

    def betweenness_centrality(self):
        """Tính độ trung tâm trung gian (chuẩn hóa) cho mỗi đỉnh"""
        if self.num_vertices <= 2:
            return {v: 0 for v in self.__adjacency_list}
        
        # Hệ số chuẩn hóa
        normalize = (self.num_vertices - 1) * (self.num_vertices - 2)
        if not self.__directed:
            normalize /= 2
        
        betweenness = {v: 0 for v in self.__adjacency_list}
        
        for s in self.__adjacency_list:
            distances, predecessors = self._shortest_paths_bfs(s)
            
            for t in self.__adjacency_list:
                if t == s:
                    continue
                if distances[t] != float('inf'):
                    stack = [(t, 1.0)]
                    while stack:
                        vertex, path_count = stack.pop()
                        for pred in predecessors[vertex]:
                            if pred != s:
                                betweenness[pred] += path_count
                            if pred != s:
                                stack.append((pred, path_count))
        
        for v in betweenness:
            betweenness[v] /= normalize
        
        return betweenness

    def closeness_centrality(self):
        """Tính độ trung tâm gần kề (chuẩn hóa) cho mỗi đỉnh"""
        closeness = {}
        
        for vertex in self.__adjacency_list:
            distances, _ = self._shortest_paths_bfs(vertex)
            
            total_distance = sum(1/d for d in distances.values() if d != float('inf') and d != 0)
            reachable_nodes = sum(1 for d in distances.values() if d != float('inf')) - 1
            
            if reachable_nodes > 0:
                closeness[vertex] = total_distance * (reachable_nodes / (self.num_vertices - 1))
            else:
                closeness[vertex] = 0
                
        return closeness

    def clustering_coefficient(self):
        """Tính hệ số phân cụm cho mỗi đỉnh và toàn bộ đồ thị"""
        local_clustering = {}
        
        for vertex in self.__adjacency_list:
            neighbors = set(self.__adjacency_list[vertex])
            if len(neighbors) < 2:
                local_clustering[vertex] = 0
                continue
            
            # Đếm số cạnh giữa các láng giềng
            edges_between_neighbors = 0
            for u in neighbors:
                for v in neighbors:
                    if u != v:
                        if self.__directed:
                            if v in self.__adjacency_list[u]:
                                edges_between_neighbors += 1
                        else:
                            if v in self.__adjacency_list[u]:
                                edges_between_neighbors += 0.5  # Tránh đếm 2 lần trong đồ thị vô hướng
            
            # Tính hệ số phân cụm cục bộ
            if self.__directed:
                max_possible_edges = len(neighbors) * (len(neighbors) - 1)
            else:
                max_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                
            local_clustering[vertex] = edges_between_neighbors / max_possible_edges if max_possible_edges > 0 else 0
        
        # Tính hệ số phân cụm toàn cục (trung bình của các hệ số cục bộ)
        global_clustering = sum(local_clustering.values()) / len(local_clustering) if local_clustering else 0
        
        return {
            'local': local_clustering,
            'global': global_clustering
        }
    def similarity_matrix(self):
        """Tính ma trận similarity trung gian của đồ thị"""
        # Khởi tạo ma trận similarity với giá trị 0
        similarity = {v: {u: 0 for u in self.__adjacency_list} for v in self.__adjacency_list}
        
        # Tính độ tương đồng cho từng cặp đỉnh
        for v in self.__adjacency_list:
            for u in self.__adjacency_list:
                # Lấy tập hợp láng giềng của v và u
                neighbors_v = set(self.__adjacency_list[v])
                neighbors_u = set(self.__adjacency_list[u])
                
                # Tìm giao của hai tập láng giềng
                common_neighbors = neighbors_v.intersection(neighbors_u)
                
                # Tính tổng 1/deg(z) cho mỗi đỉnh z trong tập giao
                similarity[v][u] = sum(1/len(self.__adjacency_list[z]) for z in common_neighbors)
        
        return similarity
    

# Subclass WeightedGraph from Graph (weighted graph)
class WeightedGraph(Graph):
    def __init__(self, directed=False):
        """Initialize the weighted graph with direction option"""
        super().__init__(directed)
        self.__adjacency_list = {}  # Override with private attribute specific to WeightedGraph
    
    def add_edge(self, u, v, weight=1):
        """Override add_edge method to add weight for the edge"""
        if u not in self.__adjacency_list:
            self.__adjacency_list[u] = []
        if v not in self.__adjacency_list:
            self.__adjacency_list[v] = []

        # Add edge with weight to the adjacency list
        self.__adjacency_list[u].append((v, weight))
        if not self.is_directed():
            self.__adjacency_list[v].append((u, weight))  # Add reverse edge if undirected
    
    def get_edges(self):
        """Return a list of all edges in the graph with weights"""
        edges = []
        for u in self.__adjacency_list:
            for v, weight in self.__adjacency_list[u]:
                # For undirected graphs, ensure that we don't add reverse edges twice
                if self.is_directed() or (u, v, weight) not in edges and (v, u, weight) not in edges:
                    edges.append((u, v, weight))
        return edges


    def dijkstra(self, start_vertex):
        """Dijkstra's algorithm for shortest path using custom PriorityQueue"""
        if start_vertex not in self.__adjacency_list:
            raise VertexNotFoundException(start_vertex)
        
        # Initialize distances: all distances set to infinity
        dist = {vertex: float('inf') for vertex in self.__adjacency_list}
        dist[start_vertex] = 0
        
        # Initialize PriorityQueue and add the start vertex with distance 0
        priority_queue = PriorityQueue()
        priority_queue.enqueue((start_vertex, 0), 0)
        
        prev = {vertex: None for vertex in self.__adjacency_list}
        
        while not priority_queue.is_empty():
            (current_vertex, current_dist), _ = priority_queue.dequeue()
            
            # Skip if we found a better path already
            if current_dist > dist[current_vertex]:
                continue
            
            # Check all neighbors of the current vertex
            for neighbor, weight in self.__adjacency_list[current_vertex]:
                distance = dist[current_vertex] + weight
                
                # If a shorter path to the neighbor is found
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current_vertex
                    priority_queue.enqueue((neighbor, distance), distance)
        
        return dist, prev



    def bellman_ford(self, start_vertex):
        """Bellman-Ford algorithm for shortest path and detecting negative cycles"""
        adjacency_list = self.get_adjacency_list()
        dist = {vertex: float('inf') for vertex in adjacency_list}
        dist[start_vertex] = 0
        
        for _ in range(len(adjacency_list) - 1):
            for u, v, weight in self.get_edges():
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
        
        # Detect negative weight cycles
        for u, v, weight in self.get_edges():
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                raise ValueError("Graph contains a negative weight cycle")
        
        return dist

    def floyd_warshall(self):
        """Floyd-Warshall algorithm for all-pairs shortest path"""
        vertices = self.get_adjacency_list().keys()
        dist = {u: {v: float('inf') for v in vertices} for u in vertices}
        
        for u in vertices:
            dist[u][u] = 0
        
        for u, v, weight in self.get_edges():
            dist[u][v] = weight
            if not self.is_directed():
                dist[v][u] = weight
        
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist

    def prim(self):
        """Prim's algorithm for Minimum Spanning Tree using custom PriorityQueue"""
        adjacency_list = self.get_adjacency_list()
        
        if not adjacency_list:
            return []  # If the graph is empty, return an empty MST
        
        start_vertex = next(iter(adjacency_list))  # Start from an arbitrary vertex
        visited = set([start_vertex])  # Set of visited vertices
        
        priority_queue = PriorityQueue()
        
        # Push all edges from the start vertex into the priority queue
        for v, weight in adjacency_list[start_vertex]:
            priority_queue.enqueue((start_vertex, v), weight)
        
        mst = []  # List to store the edges of the Minimum Spanning Tree (MST)
        
        while not priority_queue.is_empty():
            (u, v), weight = priority_queue.dequeue()  # Get the edge with smallest weight
            
            # If the vertex 'v' is already visited, skip it
            if v in visited:
                continue
            
            # Otherwise, add the edge (u, v) to the MST and mark 'v' as visited
            visited.add(v)
            mst.append((u, v, weight))
            
            # Add all the edges from vertex 'v' to the priority queue
            for to_neighbor, edge_weight in adjacency_list[v]:
                if to_neighbor not in visited:
                    priority_queue.enqueue((v, to_neighbor), edge_weight)
        
        return mst


    def kruskal(self):
        """Kruskal's algorithm for Minimum Spanning Tree using Union-Find"""
        
        # Initialize Union-Find for all vertices in the graph
        vertices = list(self.get_adjacency_list().keys())  # Get all vertices
        uf = UnionFind(len(vertices))  # Union-Find structure initialized with the number of vertices
        
        # Mapping from vertex to index for Union-Find operations
        vertex_to_index = {vertex: i for i, vertex in enumerate(vertices)}
        
        # Get all edges in the graph and sort them by weight
        edges = self.get_edges()  # Get all edges from the adjacency list
        edges.sort(key=lambda edge: edge[2])  # Sort edges by weight
        
        mst = []  # List to store the edges of the Minimum Spanning Tree (MST)
        
        # Process edges in increasing order of weight
        for u, v, weight in edges:
            # Find the sets (roots) of u and v using Union-Find
            root_u = uf.find(vertex_to_index[u])
            root_v = uf.find(vertex_to_index[v])
            
            # If u and v are in different sets, add the edge to the MST
            if root_u != root_v:
                mst.append((u, v, weight))
                uf.union(root_u, root_v)  # Union the sets of u and v
        
        return mst

def main_undirected_graphs():
    # --- Test Graph class (unweighted) ---
    print("=== Testing Graph (Unweighted) ===")
    graph = Graph(directed=False)
    graph.add_vertex("A")
    graph.add_vertex("B")
    graph.add_vertex("C")
    graph.add_vertex("D")

    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")

    # BFS from A
    print("BFS from A:", graph.bfs("A"))

    # DFS from A
    print("DFS from A:", graph.dfs("A"))

    # Beam Search from A with beam width 2
    print("Beam Search from A with beam width 2:", graph.beam_search("A", 2))

    # Remove an edge and test again
    graph.remove_edge("A", "C")
    print("BFS after removing edge A-C:", graph.bfs("A"))

    # --- Test WeightedGraph class (weighted) ---
    print("\n=== Testing WeightedGraph (Weighted) ===")
    weighted_graph = WeightedGraph(directed=False)
    weighted_graph.add_vertex("A")
    weighted_graph.add_vertex("B")
    weighted_graph.add_vertex("C")
    weighted_graph.add_vertex("D")

    weighted_graph.add_edge("A", "B", 1)
    weighted_graph.add_edge("A", "C", 4)
    weighted_graph.add_edge("B", "C", 2)
    weighted_graph.add_edge("B", "D", 3)
    weighted_graph.add_edge("C", "D", 5)

    # Get all edges in the weighted graph
    print("All edges in the weighted graph:", weighted_graph.get_edges())

    # Dijkstra's algorithm from A
    dist, prev = weighted_graph.dijkstra("A")
    print("Dijkstra's shortest paths from A:", dist)

    # Bellman-Ford algorithm from A
    try:
        bellman_dist = weighted_graph.bellman_ford("A")
        print("Bellman-Ford shortest paths from A:", bellman_dist)
    except ValueError as e:
        print(e)

    # Floyd-Warshall algorithm (all-pairs shortest paths)
    floyd_dist = weighted_graph.floyd_warshall()
    print("Floyd-Warshall all-pairs shortest paths:")
    for u, v_dict in floyd_dist.items():
        print(f"{u}: {v_dict}")

    # Prim's algorithm (MST)
    prim_mst = weighted_graph.prim()
    print("Prim's algorithm (MST):", prim_mst)

    # Kruskal's algorithm (MST)
    kruskal_mst = weighted_graph.kruskal()
    print("Kruskal's algorithm (MST):", kruskal_mst)

def main_directed_graphs():
    # --- Test Directed Graph (Unweighted) ---
    print("=== Testing Directed Graph (Unweighted) ===")
    directed_graph = Graph(directed=True)
    directed_graph.add_vertex("A")
    directed_graph.add_vertex("B")
    directed_graph.add_vertex("C")
    directed_graph.add_vertex("D")

    directed_graph.add_edge("A", "B")
    directed_graph.add_edge("A", "C")
    directed_graph.add_edge("B", "D")
    directed_graph.add_edge("C", "D")

    # BFS from A
    print("BFS from A:", directed_graph.bfs("A"))

    # DFS from A
    print("DFS from A:", directed_graph.dfs("A"))

    # Beam Search from A with beam width 2
    print("Beam Search from A with beam width 2:", directed_graph.beam_search("A", 2))

    # Remove an edge and test again
    directed_graph.remove_edge("A", "C")
    print("BFS after removing edge A-C:", directed_graph.bfs("A"))

    # --- Test Directed Weighted Graph (Weighted) ---
    print("\n=== Testing Directed Weighted Graph (Weighted) ===")
    directed_weighted_graph = WeightedGraph(directed=True)
    directed_weighted_graph.add_vertex("A")
    directed_weighted_graph.add_vertex("B")
    directed_weighted_graph.add_vertex("C")
    directed_weighted_graph.add_vertex("D")

    directed_weighted_graph.add_edge("A", "B", 1)
    directed_weighted_graph.add_edge("A", "C", 4)
    directed_weighted_graph.add_edge("B", "C", 2)
    directed_weighted_graph.add_edge("B", "D", 3)
    directed_weighted_graph.add_edge("C", "D", 5)

    # Get all edges in the directed weighted graph
    print("All edges in the directed weighted graph:", directed_weighted_graph.get_edges())

    # Dijkstra's algorithm from A
    dist, prev = directed_weighted_graph.dijkstra("A")
    print("Dijkstra's shortest paths from A:", dist)

    # Bellman-Ford algorithm from A
    try:
        bellman_dist = directed_weighted_graph.bellman_ford("A")
        print("Bellman-Ford shortest paths from A:", bellman_dist)
    except ValueError as e:
        print(e)

    # Floyd-Warshall algorithm (all-pairs shortest paths)
    floyd_dist = directed_weighted_graph.floyd_warshall()
    print("Floyd-Warshall all-pairs shortest paths:")
    for u, v_dict in floyd_dist.items():
        print(f"{u}: {v_dict}")

    # Prim's algorithm (MST) - Prim không phù hợp với đồ thị có hướng
    print("Prim's algorithm không được áp dụng cho đồ thị có hướng.")

    # Kruskal's algorithm (MST) - Kruskal không phù hợp với đồ thị có hướng
    print("Kruskal's algorithm không được áp dụng cho đồ thị có hướng.")

def main_metrics():
    print("\n=== Testing Graph Metrics ===")
    g = Graph(directed=False)
    
    # Thêm đỉnh và cạnh
    vertices = ["A", "B", "C", "D", "E"]
    for v in vertices:
        g.add_vertex(v)
    
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("D", "E")
    g.add_edge("E", "A")
    g.add_edge("A", "C")
    
    print("Density:", g.density())
    print("Degree Centrality:", g.degree_centrality())
    print("Betweenness Centrality:", g.betweenness_centrality())
    print("Closeness Centrality:", g.closeness_centrality())
    print("Clustering Coefficient:", g.clustering_coefficient())
    print("Similarity Matrix:", g.similarity_matrix())
if __name__ == "__main__":
    main_undirected_graphs()
    main_directed_graphs()
    main_metrics()






