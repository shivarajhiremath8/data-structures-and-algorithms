# =======================
# 1️⃣ Basics
# =======================
x = 5
y = 3.2
name = "Python"
print("Sum:", x + y)
print("Type of name:", type(name))

# =======================
# 2️⃣ Conditional Statements
# =======================
num = 10
if num % 2 == 0:
    print("Even")
else:
    print("Odd")

# =======================
# 3️⃣ Loops
# =======================
for i in range(1, 6):
    print("Loop:", i)

count = 0
while count < 3:
    print("Count:", count)
    count += 1

# =======================
# 4️⃣ Functions
# =======================
def greet(user):
    return f"Hello, {user}!"

print(greet("Raju"))

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print("Factorial of 5:", factorial(5))

# =======================
# 5️⃣ Lists
# =======================
nums = [1, 2, 3, 4, 5]
nums.append(6)
nums.remove(3)
print("List:", nums)
print("Sliced:", nums[1:4])
squares = [i**2 for i in nums]
print("Squares:", squares)

# =======================
# 6️⃣ Tuples
# =======================
coords = (10, 20)
print("X:", coords[0], "Y:", coords[1])

# =======================
# 7️⃣ Sets
# =======================
a = {1, 2, 3}
b = {3, 4, 5}
print("Union:", a | b)
print("Intersection:", a & b)

# =======================
# 8️⃣ Dictionaries
# =======================
student = {"name": "Raju", "age": 20, "course": "Python"}
print(student["name"])
student["age"] = 21
for key, val in student.items():
    print(key, ":", val)

# =======================
# 9️⃣ Strings
# =======================
text = "hello python"
print("Upper:", text.upper())
print("Split:", text.split())
print("Reverse:", text[::-1])
print("Count 'l':", text.count("l"))

# =======================
# 🔟 File Handling
# =======================
with open("demo.txt", "w") as f:
    f.write("Learning Python is fun!")

with open("demo.txt", "r") as f:
    content = f.read()
    print("File Content:", content)

# =======================
# 1️⃣1️⃣ Lambda & Map
# =======================
nums2 = [1, 2, 3, 4]
doubled = list(map(lambda n: n * 2, nums2))
print("Doubled:", doubled)

# =======================
# 1️⃣2️⃣ Object-Oriented Programming
# =======================
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def display(self):
        print(f"Car: {self.brand} {self.model}")

car1 = Car("Toyota", "Supra")
car1.display()

# Inheritance
class ElectricCar(Car):
    def __init__(self, brand, model, battery):
        super().__init__(brand, model)
        self.battery = battery

ecar = ElectricCar("Tesla", "Model 3", "85kWh")
ecar.display()
print("Battery:", ecar.battery)

# =======================
# 1️⃣3️⃣ Exception Handling
# =======================
try:
    val = int("abc")
except ValueError:
    print("Invalid conversion!")

# =======================
# 1️⃣4️⃣ Recursion Example
# =======================
def sum_to_n(n):
    if n == 0:
        return 0
    return n + sum_to_n(n - 1)

print("Sum to 10:", sum_to_n(10))

# =======================
# 1️⃣5️⃣ Modules Example
# =======================
import math, random, datetime

print("Square root of 16:", math.sqrt(16))
print("Random number:", random.randint(1, 100))
print("Current time:", datetime.datetime.now())
