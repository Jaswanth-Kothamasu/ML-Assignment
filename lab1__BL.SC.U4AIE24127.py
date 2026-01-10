#Question 1
"""
def que1(numbers,target_sum):
    count=0
    length=len(numbers)
    for i in range(length):
        for j in range(i+1,length):
            if numbers[i]+numbers[j]==target_sum:
                count+=1
    return count

numbers_list=[2,7,4,1,3,6]
result=que1(numbers_list,10)
print("Number of pairs with sum 10 :",result)
"""

#Question 2
"""
def range(values):
    if len(values)<3:
        return "Range determination not possible"
    return max(values)-min(values)

input_list=[5,3,8,1,0,4]
range_value = range(input_list)
print("Range of the list:", range_value)
"""

#Question 3
"""
import numpy as np
def matrix_power_numpy(matrix, power):
    result = np.array(matrix)
    for _ in range(power-1):
        result = np.dot(result,matrix)
    return result
matrix_A=[
    [1,2],
    [3,4]
]
m=2
power_result = matrix_power_numpy(matrix_A,m)
print("Matrix A raised to power",m)
print(power_result)

"""

#Question 4
"""
def highest_occurring_character(text):
    maximum_count=0
    maximum_character=''
    for character in text:
        if character.isalpha():
            count = text.count(character)
            if count>maximum_count:
                maximum_count=count
                maximum_character=character
    return maximum_character,maximum_count

input_string="hippopotamus"
character,count=highest_occurring_character(input_string)

print("Highest occurring character:", character)
print("Occurrence count:", count)

"""

#Question 5
"""
import random
def calculate_mean(numbers):
    return sum(numbers)/len(numbers)
def calculate_median(numbers):
    numbers.sort()
    return numbers[len(numbers)//2]
def calculate_mode(numbers):
    return max(numbers,key=numbers.count)

random_numbers = []
for _ in range(25):
    random_numbers.append(random.randint(1, 10))

mean_value = calculate_mean(random_numbers)
median_value = calculate_median(random_numbers)
mode_value = calculate_mode(random_numbers)

print("Random Numbers:", random_numbers)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)

"""







