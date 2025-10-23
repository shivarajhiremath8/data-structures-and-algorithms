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
