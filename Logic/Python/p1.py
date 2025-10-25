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
