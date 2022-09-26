import copy

# Arithmetic operators
print(1 + 1)
print(1 - 1)
print(5 * 10)
print(10/5)
print(((6+9)*3)+3)
print(6 % 2)
print(6 ^ 2)
print(6 ** 2)
print(13 // 2)

# Variables - snake_case
age = 25

# Integer are immutable
a = 1
b = a
a = 'Hello, world'

a = 1
b = a
a = a - 1

# Lists are mutable
list_1 = [1, 2, 3]
list_2 = list_1
list_2 = [4, 5, 6]
print(list_1)
print(list_2)

list_1 = [1, 2, 3]
list_2 = list_1
list_1[0] = -1
print(list_2)

a = [1, 3, 5, 7]
b = copy.copy(a)
b[0] = -1
print(a)
print(b)

a = [1, 3, 5, 7]
b = [2, 4, 6, 8]
c = [a, b]
d = copy.copy(c)
a[0] = -1
c[0][1] = -3
print(d)

a = [1, 3, 5, 7]
b = [2, 4, 6, 8]
c = [a, b]
d = copy.deepcopy(c)
a[0] = -1
c[0][1] = -3
print(d)

# Equality of object ==, is
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)
print(a == c)
print(a is b)
print(a is c)

# Problems with dynamically typed languages
a = [1, 2, 3]
a = 1

#a[0] = 1

# Variables
age, name, gender = 77, "John", "Male"
blood_type = 'AB'


# region Numeric data types
bank_balance = 500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
print(bank_balance)

x = 7
y = 3
z = x/y
print(z)
print(type(z))

w = int(3.7)
some_value = float(3)
# endregion

# region Strings
name = "Tadas"
print(name)
print(type(name))

dialogue = 'Luke said, "Hello, World"'
dialogue = 'Luke said, "Hello, World", to which the world responded with "You\'re the best Luke"'

segment_one = "I\'m getting old"
segment_two = " yes?"
full_sentence = segment_one + segment_two

print(len(full_sentence*10))
# endregion

# region Booleans
this_course_is_the_best = True
this_course_is_not_the_best = False

comparison = 1 > 2
print(comparison)

print((1 < 2) or (2 < 3))
print((1 < 2) or (2 > 3))
print((1 < 2) and (2 > 3))
print(not(1 < 2))


# endregion

# region Methods
movie_title = "Nope"
print(movie_title.upper())
print(movie_title.count('a'))
# endregion

# region Collections - Lists
names = ["Gemma", "Luke", "Tadas"]
print(names[0])
# print(names[100])
random_variables = [True, False, "Hello", 1, 1.2]
print(len(random_variables))
length = len(random_variables)
print(random_variables[-1])

# Slicing
ordered_numbers = list(range(0, 11))
print(ordered_numbers)
print(ordered_numbers[0:5])
print(ordered_numbers[:7])
print(ordered_numbers[2:])

every_fifth = list(range(0, 101, 5))
print(every_fifth)

months = ["Jan", "Feb", "Mar"]
print("Jan" in months)
print("Jun" in months)

course = "Smart Tech"
print("art" in course)
# endregion

numbers = [1, 2, 3, 4]
print(len(numbers))
print(max(numbers))

the_very_bad_list = ["Tadas", "Luke", "Liam", "Richard", "tadas"]
print(max(the_very_bad_list))
print(sorted(the_very_bad_list))

print('-'.join(['Jan', 'Feb', 'Mar']))

print('The person is {}, {}, and {}'.format('tall', 'thin', 'old'))

months = ['Jan', 'Feb', 'Mar']
months.append('Apr')

# region Tuples
# Immutable, ordered
traits = ('tall', 'thin', 23, 'Dundalk', 1.73)
height_desc, _, age, home_town, height = traits

# endregion

# region Sets
# Mutable, unordered, unique elements
duplicates_numbers = [1, 1, 2, 2, 3, 3]
unique_numbers = set(duplicates_numbers)
print(unique_numbers)
unique_numbers.add(4)
unique_numbers.add(3)
print(3 in unique_numbers)
# endregion

# region Dictionaries
# Mutable, Not ordered
# keys MUST be unique and immutable
inventory = {'bananas': 1.30, 'apples': 0.99, 'grapes': '2.99'}
print(inventory['bananas'])
inventory['bananas'] = 1.49
print(inventory['bananas'])
bananas_price = inventory.get('bananas')
strawberries_price = inventory.get('strawberries')
inventory['apples'] = None
print(inventory['apples'])
print(strawberries_price)

print('apples' in inventory)

grocery_items = {'bananas': {'price': 1.49, 'country_of_origin': 'Cavan'}}
print(grocery_items['bananas'])
print(grocery_items['bananas']['country_of_origin'])
# endregion

# region Control Structures
inventory = {'bananas': 1.30, 'apples': 0.99, 'grapes': '2.99'}
item = 'cabbage'
if item in inventory:
    print("Found the", item)
else:
    print("Not found")
    inventory.update({item: 0.79})
    print("Just added the item. Here is the updated list", inventory)

months = ['Jan', 'Feb', 'Mar']
for month in months:
    print(month)

for number in range(0, 100):
    print(number)

names = ['denis', 'ethan sia', 'samuel']
for index in range(len(names)):
    print(index)
    names[index] = names[index].capitalize()

print(names)

movies = {'Interstellar': 2014, 'Tron Legacy': 2010, 'Magic Mic': 2012}
for key in movies:
    print(key)

for key, value in movies.items():
    print(key, value)

# The movie Star Wars was released in ...
for key, value in movies.items():
    print(f"The movie {key}, was released in {value}")

# while
number = 20
while number < 31:
    print(number)
    number += 1

numbers = list(range(10))
print(numbers)
for number in numbers:
    if number % 2 != 0:
        break
    print(number)


# endregion

# region Functions
def random_function():
    random_name = 'Tolani'

#print(random_name)

# Lambda and higher order functions
numbers = [1, 2, 3, 4, 5]


def even_or_odd(num):
    return num % 2 == 0

# Higher order functions take functions as args - see filter

#print(list(filter(even_or_odd, numbers)))

# Lambda = anonymous function, used

print(list(filter(lambda num: num % 2 == 0, numbers)))

list_two = list(range(1, 4))
list_three = list(range(1, 4))
list_sum = []

# list_sum = list_two**2+list_three**3

for index in range(3):
    list_two[index] = list_two[index]**2
    list_three[index] = list_three[index]**3
    list_sum.append(list_two[index]+list_three[index])

print(list_sum)
# endregion

# region Numpy
import numpy as np

array_two = np.arange(1, 4)**2
print(array_two)
array_three = np.arange(1, 4)**3
print(array_two + array_three)

print(np.power(np.array([1, 2, 3, 4]), 4))
print(np.negative(np.array([1, 2, 3, 4])))

sample_array = np.array([1, 2, 3, 4])
print(np.exp(sample_array))
print(np.log(sample_array))
print(np.sin(sample_array))

print(sample_array.shape)

x = np.arange(3)
y = np.arange(3)
z = np.arange(3)

multi_array = np.array([x, y, z])
print(multi_array)
print(multi_array.shape)

aw = np.arange(1, 10, 2)
print(aw)
w = np.linspace(1, 10, 50)
print(w)

b = np.arange(1, 30, 3)
print(b)
b = np.linspace(1, 30, 3, False)
print(b)

x = np.arange(3)
y = np.arange(3, 6)
z = np.arange(6, 9)

multi_array = np.array([x, y, z], dtype=np.uint8)
print(multi_array)

print(multi_array[1][2])

print(multi_array.dtype)

# Slicing
# [start stop step]
x = np.arange(1, 10)
print(x[2:])

x = np.arange(18).reshape(3, 2, 3)
print(x[1, ...])










# endregion



