# =======================
# 1️⃣ ARRAYS / LISTS
# =======================
nums = [5, 3, 8, 1, 2]
nums.sort()
print("Sorted array:", nums)

# Find max and min
print("Max:", max(nums), "Min:", min(nums))

# Reverse array
nums.reverse()
print("Reversed:", nums)

# Linear search
target = 8
for i, val in enumerate(nums):
    if val == target:
        print("Found at index:", i)
        break

# =======================
# 2️⃣ STRINGS
# =======================
text = "level"
print("Is palindrome:", text == text[::-1])

# Count vowels
vowels = "aeiou"
count = sum(1 for c in text if c in vowels)
print("Vowel count:", count)

# Frequency of characters
from collections import Counter
print("Char freq:", Counter("banana"))

# =======================
# 3️⃣ STACK (LIFO)
# =======================
stack = []
stack.append(10)
stack.append(20)
stack.append(30)
print("Stack before pop:", stack)
stack.pop()
print("Stack after pop:", stack)

# =======================
# 4️⃣ QUEUE (FIFO)
# =======================
from collections import deque
queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
print("Queue before popleft:", list(queue))
queue.popleft()
print("Queue after popleft:", list(queue))

# =======================
# 5️⃣ LINKED LIST
# =======================
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        newnode = Node(data)
        if not self.head:
            self.head = newnode
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = newnode

    def display(self):
        curr = self.head
        while curr:
            print(curr.data, end=" -> ")
            curr = curr.next
        print("None")

ll = LinkedList()
for val in [1, 2, 3, 4]:
    ll.append(val)
ll.display()

# =======================
# 6️⃣ RECURSION
# =======================
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print("Factorial(5):", factorial(5))

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print("Fibonacci(6):", fibonacci(6))

# =======================
# 7️⃣ BINARY SEARCH
# =======================
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

sorted_nums = [1, 3, 5, 7, 9, 11]
print("Index of 7:", binary_search(sorted_nums, 7))

# =======================
# 8️⃣ SORTING (BUBBLE SORT)
# =======================
arr = [5, 2, 9, 1, 5, 6]
for i in range(len(arr)):
    for j in range(0, len(arr) - i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
print("Bubble sorted:", arr)

# =======================
# 9️⃣ HASHING (USING DICT)
# =======================
nums_list = [1, 2, 3, 2, 1, 4]
freq = {}
for num in nums_list:
    freq[num] = freq.get(num, 0) + 1
print("Frequency map:", freq)

# =======================
# 🔟 MERGE SORT (DIVIDE & CONQUER)
# =======================
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

nums_merge = [7, 3, 1, 9, 5, 8]
print("Merge sorted:", merge_sort(nums_merge))

# =======================
# 1️⃣1️⃣ STACK VALID PARENTHESIS (Classic)
# =======================
def is_valid_parentheses(s):
    pair = {')': '(', '}': '{', ']': '['}
    st = []
    for ch in s:
        if ch in pair.values():
            st.append(ch)
        elif ch in pair:
            if not st or st.pop() != pair[ch]:
                return False
    return not st

print("Valid parentheses:", is_valid_parentheses("({[]})"))

# =======================
# 1️⃣2️⃣ PREFIX SUM (ARRAY)
# =======================
arr_sum = [2, 4, 6, 8, 10]
prefix = [0]
for n in arr_sum:
    prefix.append(prefix[-1] + n)
print("Prefix sums:", prefix)
print("Sum of range(1,3):", prefix[3] - prefix[1])

# =======================
# 1️⃣3️⃣ TWO POINTERS (ARRAY)
# =======================
def two_sum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return (left, right)
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return None
sorted_arr = [1, 2, 3, 4, 6, 8]
print("Two sum indices for 10:", two_sum(sorted_arr, 10))
# =======================
# 1️⃣4️⃣ DEPTH FIRST SEARCH (DFS) (GRAPH)
# =======================
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
graph_example = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0],
    3: [1],
    4: [1, 5],
    5: [4]
}
print("DFS starting from node 0:", dfs(graph_example, 0))
# =======================
# 1️⃣5️⃣ BREADTH FIRST SEARCH (BFS) (GRAPH)
# =======================
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
print("BFS starting from node 0:", bfs(graph_example, 0))
# =======================
# 1️⃣6️⃣ DIJKSTRA'S ALGORITHM (SHORTEST PATH) (GRAPH)
# =======================
import heapq
def dijkstra(graph, start):
    min_heap = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    while min_heap:
        curr_dist, curr_node = heapq.heappop(min_heap)
        if curr_dist > distances[curr_node]:
            continue
        for neighbor, weight in graph[curr_node].items():
            distance = curr_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(min_heap, (distance, neighbor))
    return distances
graph_weights = {
    0: {1: 4, 2: 1},
    1: {3: 1},
    2: {1: 2, 3: 5},
    3: {}
}
print("Dijkstra's shortest paths from node 0:", dijkstra(graph_weights, 0
))
# =======================
# 1️⃣7️⃣ TOPOLOGICAL SORT (DAG)
# =======================
def topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    queue = deque([u for u in graph if in_degree[u] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return topo_order
dag_example = {
    5: [2, 0],
    4: [0, 1],
    3: [1],
    2: [3],
    1: [],
    0: []
}
print("Topological sort of DAG:", topological_sort(dag_example))
# =======================
# 1️⃣8️⃣ UNION-FIND (DISJOINT SET)
# =======================
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1
uf = UnionFind(10)
uf.union(1, 2)
uf.union(2, 3)
print("Find 3:", uf.find(3))
print("Find 4:", uf.find(4))
# =======================
# 1️⃣9️⃣ BINARY TREE TRAVERSALS
# =======================
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def inorder_traversal(root):
    return inorder_traversal(root.left) + [root.val] + inorder_traversal(root.right) if root else []
def preorder_traversal(root):
    return [root.val] + preorder_traversal(root.left) + preorder_traversal(root.right) if root else []
def postorder_traversal(root):
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.val] if root else []
# Example tree
root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)
print("Inorder:", inorder_traversal(root))
print("Preorder:", preorder_traversal(root))
print("Postorder:", postorder_traversal(root))
# =======================
# 2️⃣0️⃣ HEAP (PRIORITY QUEUE)
# =======================
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
print("Heap after pushes:", heap)
min_elem = heapq.heappop(heap)
print("Popped min element:", min_elem)
print("Heap after pop:", heap)
# =======================
# 2️⃣1️⃣ GREEDY ALGORITHM (ACTIVITY SELECTION)
# =======================
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])
    selected = []
    last_end_time = 0
    for start, end in activities:
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    return selected
activities = [(1, 3), (2, 5), (4, 6), (6, 7), (5, 8)]
print("Selected activities:", activity_selection(activities))
# =======================
# 2️⃣2️⃣ KMP ALGORITHM (STRING MATCHING)
# =======================
def kmp_search(text, pattern):
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1
print("KMP search index:", kmp_search("ababcababcabc", "abc"))

# =======================
# 2️⃣3️⃣ AVL TREE (SELF-BALANCING BINARY SEARCH TREE)
# =======================
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1
class AVLTree:
    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        return root
    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y
    def right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y
    def get_height(self, root):
        if not root:
            return 0
        return root.height
    def get_balance(self, root):
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)
avl = AVLTree()
root = None
for key in [10, 20, 30, 40, 50, 25]:
    root = avl.insert(root, key)
print("AVL Tree root after insertions:", root.key)

# =======================
# 2️⃣4️⃣ FORD-FULKERSON ALGORITHM (MAX FLOW)
# =======================

from collections import defaultdict
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(dict)

    def add_edge(self, u, v, w):
        self.graph[u][v] = w

    def bfs(self, s, t, parent):
        visited = [False] * self.V
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if not visited[v] and self.graph[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == t:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.V
        max_flow = 0
        while self.bfs(source, sink, parent):
            path_flow = float('Inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] = self.graph.get(v, {}).get(u, 0) + path_flow
                v = parent[v]
            max_flow += path_flow
        return max_flow
g = Graph(6)
g.add_edge(0, 1, 16)
g.add_edge(0, 2, 13)
g.add_edge(1, 2, 10)
g.add_edge(1, 3, 12)
g.add_edge(2, 1, 4)
g.add_edge(2, 4, 14)
g.add_edge(3, 2, 9)
g.add_edge(3, 5, 20)
g.add_edge(4, 3, 7)
g.add_edge(4, 5, 4)
source, sink = 0, 5
print("The maximum possible flow is:", g.ford_fulkerson(source, sink))

# =======================
# 2️⃣5️⃣ TRIE (PREFIX TREE)
# =======================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
trie = Trie()
trie.insert("hello")
trie.insert("world")
print("Search 'hello':", trie.search("hello"))
print("Search 'python':", trie.search("python"))

# =======================
# 2️⃣6️⃣ SEGMENT TREE (RANGE QUERY)
# =======================

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)

    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[i * 2] + self.tree[i * 2 + 1]

    def update(self, index, value):
        pos = index + self.n
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.tree[pos * 2] + self.tree[pos * 2 + 1]

    def range_query(self, left, right):
        result = 0
        left += self.n
        right += self.n
        while left < right:
            if left % 2:
                result += self.tree[left]
                left += 1
            if right % 2:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result
data = [1, 3, 5, 7, 9, 11]
seg_tree = SegmentTree(data)
print("Range sum (1,4):", seg_tree.range_query(1, 4))
seg_tree.update(1, 10)
print("Range sum (1,4) after update:", seg_tree.range_query(1, 4))

# =======================
# 2️⃣7️⃣ BINARY HEAP (MIN HEAP)
# =======================

class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, key):
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._heapify_up(parent)

    def _heapify_down(self, index):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)
min_heap = MinHeap()
for num in [5, 3, 8, 1, 2]:
    min_heap.insert(num)
print("Extracted min:", min_heap.extract_min())
print("Extracted min:", min_heap.extract_min())
print("Extracted min:", min_heap.extract_min())

# =======================
# 2️⃣8️⃣ RADIX SORT
# =======================

def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    for i in range(n):
        arr[i] = output[i]
def radix_sort(arr):
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
arr_radix = [170, 45, 75, 90, 802, 24, 2, 66]
radix_sort(arr_radix)
print("Radix sorted array:", arr_radix)

# =======================