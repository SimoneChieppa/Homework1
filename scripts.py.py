#PROBLEM 1

#INTRODUCTION
 
#Say "Hello, World!" With Python
print("Hello, World!")

#Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 != 0:
    print('Weird')
elif n in range(2,6):
    print('Not Weird')
elif n in range(6,21):
    print('Weird')
else:
    print('Not Weird')

	
#Arithmetic Operators
​
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)
	
#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)
	
#Loops

if __name__ == '__main__':
    n = int(input())
for i in range(0,n):
    print(i*i)
	

#Write a function

def is_leap(year):
    leap = False
    if year%4==0:
        leap=True
        if year%100==0:
            leap=False
            if year%400==0:
                leap = True
   

   
    
    return leap

year = int(input())
print(is_leap(year))

#Print Function

if __name__ == '__main__':
    n = int(input())
    
for i in range(1,n+1):
    print(i,end='')

	
#DATA TYPES

#List Comprehensions


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input()) 
    lista=[] 
for i in range(0,x+1):
     for j in range(0,y+1):
         for k in range(0,z+1):
            if i+j+k!=n:
                lista.append([i,j,k])
print(lista)


#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr =list(map(int,input().strip().split()))
    

max1= max(arr)
while max(arr) == max1:
    arr.remove(max(arr))

print (max(arr))


#Nested Lists

if __name__ == '__main__':
    a = []
    lowest2 = []
    scores = set()

    
    for i in range(int(input())):
        name = input()
        score = float(input())
        scores.add(score)
        a.append([name,score])


    second_lowest = sorted(scores)[1]

    for name, score in a:
        if score == second_lowest:
          lowest2.append(name)
          
    for name in sorted(lowest2):
        print(name, end='\n')
		
#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
query_mark=student_marks[query_name]
print('%.2f'%(sum(query_mark)/len(query_mark)))

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    
query_mark=student_marks[query_name]
print('%.2f'%(sum(query_mark)/len(query_mark)))


#Lists

if __name__ == '__main__':
    N = int(input())
    lista=[]
for i in range(N):
    a = input().split()
    if a[0]=="insert":
        lista.insert(int(a[1]),int(a[2]))
    elif a[0]=="remove":
        lista.remove(int(a[1]))
    elif a[0]=="append":
        lista.append(int(a[1]))
    elif a[0]=="sort":
        lista.sort()
    elif a[0]=="pop":
        lista.pop()
    elif a[0]=="reverse":
        lista.reverse()
    elif a[0]=="print":
        print(lista)
		
		
#STRINGS

#sWAP cASE

def swap_case(s):
    string=""
    for i in s:
        if i == i.upper():
            string+=i.lower()
        else:
            string+=i.upper()
            
    return string

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)
	
#String Split and Join

def split_and_join(line):
    
    line=line.split(" ")
    return "-".join(line)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?

def print_full_name(first, last):
    print(f'Hello {first} {last}! You just delved into python.')
    # Write your code here
    

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
	
#Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
	
#Find a string

def count_substring(string, sub_string):
    a = 0
    for i in range(len(string)):
        if string[i:].startswith(sub_string):
            a += 1
    return a

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
	
	
#String Validators

if __name__ == '__main__':
    s = input()

print(any(a.isalnum() for a in s))
print(any(a.isalpha()for a in s))
print(any(a.isdigit() for a in s))
print(any(a.islower() for a in s))
print(any(a.isupper()for a in s))

#Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


#Text Wrap

import textwrap

def wrap(string, max_width):

    s = ""
    for i in range(0,len(string),max_width):
        s += string[i:i+max_width] + "\n"
    return s


if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
	
#Designer Door Mat
#for this excercise I have consulted the discussion to understand how to solve it
n, m = map(int,input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))


#Capitalize!

def solve(s):
    def solve(s):
    a=""
    for i in range(len(s)):
         if i==0:
            a+=s[0].upper()
         elif s[i-1]==" ":
            a+=s[i].upper()
         else:
            a+=s[i]
    return(a)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()


#The Minion Game

def minion_game(string):
    # your code goes here
    S_score=0
    K_score=0
    vocali="AEIOU"
    for i in range(len(string)):
        if string[i] in vocali:
            K_score+= len(string)-i
        else:
            S_score+=len(string)-i
    if S_score>K_score:
        print("Stuart"+" "+ "%d" % S_score)
    elif K_score>S_score:
        print("Kevin"+" "+'%d' % K_score)
    else:
        print("Draw")

#Merge the Tools!

def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        u = ''
        for a in string[i : i+k]:
            if (a not in u):
                u+=a
        print(u)

		
#SETS

#Introduction to Sets

def average(array):
    a=set(array)
    return(sum(a)/len(a))
    
#No Idea!

_ = input()
n= input().split()
A = set(input().split())
B = set(input().split())
happiness=0
for i in range(0,len(n)):
    if n[i] in A:
        happiness+=1
    elif n[i] in B:
        happiness-=1
        
print(happiness)
    
#Symmetric Difference

M,M1=(int(input()),input().split())
N,N1=(int(input()),input().split())
a=set(M1)
b=set(N1)
p=b.difference(a)
q=a.difference(b)
r=p.union(q)
print ('\n'.join(sorted(r, key=int)))


#Set .add()

n=int(input())
countries=set()
for i in range(1,n):
    countries.add(input())
    
print(len(countries))

#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
N=int(input())
for i in range(N):
    a=input().split()
    if a[0]=="pop":
        s.pop()
    elif a[0]=="remove":
        s.remove(int(a[1]))
    elif a[0]=="discard":
        s.discard(int(a[1]))
print(sum(s))


#Set .union() Operation

_, n = input(), set(input().split())
_, b = input(), set(input().split())
s=b.union(n)
print(len(s))

#Set .intersection() Operation

_, n = input(), set(input().split())
_, b = input(), set(input().split())
s=b.intersection(n)
print(len(s))

#Set .difference() Operation

_, n = input(), set(input().split())
_, b = input(), set(input().split())

s=n.difference(b)
print(len(s))

#Set .symmetric_difference() Operation

_, n = input(), set(input().split())
_, b = input(), set(input().split())
s=n.symmetric_difference(b)
print(len(s))


#Set Mutations

n=int(input())
A=set(map(int,input().split()))
N=int(input())

for i in range(N):
    s=input().split()
    B=set(map(int,input().split()))
    if s[0]=="update":
        A.update(B)
    elif s[0]=="intersection_update":
        A.intersection_update(B)
    elif s[0]=="symmetric_difference_update":
        A.symmetric_difference_update(B)
    elif s[0]=="difference_update":
        A.difference_update(B)
print(sum(A))


#The Captain's Room

k=int(input())
lista=list(map(int, input().split()))

lista1=set(lista)
a=((sum(lista1)*k)-((sum(lista))))//(k-1)
print(a)


#Check Subset

T=int(input())
for _ in range(T):
    a=input()
    A = set(input().split())
    b= input()
    B=set(input().split())
    print(A.issubset(B))
	
#Check Strict Superset

A= set(input().split())
n=int(input())
lista = []
for i in range(n):
    B = set(input().split())
    if A.intersection(B)==B and (len(A)>len(B)) :
        lista.append(True)
    else:
         lista.append(False)
print(all(lista))


#COLLECTION

#collections.Counter()

import collections

nshoes = int(input())
shoes = collections.Counter(map(int,input().split()))
ncust = int(input())

income = 0

for i in range(ncust):
    size, price = map(int,input().split())
    if shoes[size]: 
        income += price
        shoes[size] -= 1

print (income)

#DefaultDict Tutorial

from collections import defaultdict
d = defaultdict(list)
n, m = map(int,input().split())

A=[]

for i in range(n):
    A.append(input())

    B = []
for i in range(m):
    B.append(input())

for i in range(n):
    d[A[i]].append(i+1)

for i in B:    
    if i in d:
        print(*d[i])
    else:
        print(-1)

		
#Collections.namedtuple()

from collections import namedtuple

N = int(input())
columns = input().split()

tot = 0
for i in range(N):
    students = namedtuple('student',columns)
    col1, col2, col3, col4 = input().split()
    student = students(col1, col2, col3, col4)
    tot += int(student.MARKS)
print('{:.2f}'.format(tot/N))


#Collections.OrderedDict()

from collections import OrderedDict
n=int(input())
a=OrderedDict()
for _ in range(n):
    item, space, quantity = input().rpartition(' ')
    a[item] = a.get(item, 0) + int(quantity)
for item, quantity in a.items():
    print(item, quantity)
	

#Word Order

n=int(input())
from collections import OrderedDict
a=OrderedDict()
for i in range(n):
     b = input()
     if not b in a.keys():
        a.update({b : 1})
        continue
     a[b] += 1

print(len(a.keys()))
print(*a.values())


#Collections.deque()

from collections import deque
N=int(input())
d = deque()

for i in range(N):
    a=input().split()
    if a[0]=="pop":
        d.pop()
    elif a[0]=="popleft":
        d.popleft()
    elif a[0]=="append":
        d.append(a[1])
    elif a[0]=="appendleft":
        d.appendleft(a[1])
for i in range(len(d)):
    print(d[i],end=" ")

	
#Piling Up!

for i in range(int(input())):
    n=input()
    a = list(map(int, input().split()))
    length = len(a)
    j = 0
    while j < length - 1 and a[j] >= a[j+1]:
        j += 1
    while j < length - 1 and a[j] <= a[j+1]:
        j += 1
    if  j == length - 1:
        print('Yes')
    else:
        print('No')
		
#DATE AND TIME

#Calendar Module

import calendar
m,d,y=map(int,input().split())
print((calendar.day_name[calendar.weekday(y,m,d)]).upper())


#EXCEPTIONS

N=int(input())
for i in range(N):
    a,b=(input().split())
    
   
    try:
        print(int(a)//int(b))
    except ValueError as v :
        print("Error Code:",v)
    except ZeroDivisionError:
        print("Error Code: integer division or modulo by zero")
        
#BUILT-INS

#Zipped!

N, X = map(int, input().split()) 

a= []
for _ in range(X):
    a.append( map(float, input().split()) ) 

for i in zip(*a): 
    print( sum(i)/len(i) )


#Athlete Sort

	
import math
import os
import random
import re
import sys


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

  
arr.sort(key=lambda arr:arr[k])
for j in arr:
    print(*j)


#ginortS
	
	
n = input()
l,u,o,e= [],[],[],[]

for i in n:
    if i.islower():
        l.append(i)
    elif i.isupper():
        u.append(i)
    elif i.isdigit():
        if int(i)%2 == 0:
            e.append(i)
        elif int(i)%2 != 0:
            o.append(i)
a =sorted(l)+sorted(u)+sorted(o)+sorted(e)
for i in a:
    print(i,end = "")


#PYTHON FUNCTIONALS
#Map and Lambda Function

cube = lambda x: pow(x,3) # complete the lambda function 

def fibonacci(n):
    a=[0,1]
    for i in range(2,n):
        a.append(a[i-2]+a[i-1])
    return a[0:n]
    

#REGEX AND PARSING CHALLENGES

#Detect Floating Point Number


T=int(input())
for i in range(T):

    ans=False
    try:
        string=input().strip()
        number=float(string)
        ans=True
        number=int(string)
        ans=False
    except:
        pass
    print(ans)   
	
#Validating Roman Numerals

regex_pattern = r""	# Do not delete 'r'.
thousand = "(?:(M){0,3})?"
hundred  = "(?:(D?(C){0,3})|(CM)|(CD))?"
ten      = "(?:(L?(X){0,3})|(XC)|(XL))?"
unit     = "(?:(V?(I){0,3})|(IX)|(IV))?"

regex_pattern = r"^" + thousand + hundred + ten + unit + "$"

#Validating phone numbers

import re
N=int(input())
for i in range(N):
    if re.match(r'[789]\d{9}$',input()):   
         print ('YES') 
    else:  
         print ('NO')
		 
#Validating and Parsing Email Addresses

import re
n = int(input())
for _ in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)

		
#Hex Color Code

		
import re
pattern=r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]'
for _ in range(int(input())):
    for x in re.findall(pattern,input()):
        print(x)
		
#Closures and Decorations 

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
         f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
        
        # complete the function
    return fun
	
#Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
        # complete the function
    return inner

	
#NUMPY

#Arrays

def arrays(arr):
    # complete this function
    # use numpy.array
    return numpy.array(arr, float)[::-1]
	
#Shape and Reshape

import numpy
arr = input().strip().split(' ')

a=numpy.array(arr,int)
print(numpy.reshape(a,(3,3)))


#Transpose and Flatten

import numpy

N, M = map(int, input().split())


arr = numpy.array([input().strip().split() for _ in range(N)], int)
print (arr.transpose())
print (arr.flatten())


#Concatenate

import numpy

a, b, c = map(int,input().split())
A = numpy.array([input().split() for _ in range(a)],int)
B = numpy.array([input().split() for _ in range(b)],int)
print(numpy.concatenate((A, B), axis = 0))

#Zeros and Ones

import numpy
a = tuple(  map(int,(input().split())     ))

A = numpy.zeros((a),int)
print(A)
    
B = numpy.ones((a),int)
print(B)

#Eye and Identity

import numpy

N,M= map(int, input().split())
numpy.set_printoptions(sign=' ')
print(numpy.eye(N,M))

#Array Mathematics

import numpy
N,M=map(int,input().split())
A = numpy.array([list(map(int, input().split())) for _ in range(N)], int)
B = numpy.array([list(map(int, input().split())) for _ in range(N)], int)
print(numpy.add(A,B), numpy.subtract(A,B), numpy.multiply(A,B,), numpy.floor_divide(A,B), numpy.mod(A,B), numpy.power(A,B), sep = "\n")


#Floor, Ceil and Rint

import numpy

A = numpy.array(input().split(),float)
numpy.set_printoptions(sign=' ')

print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))


#Sum and Prod

import numpy

N,M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)

print(numpy.prod(numpy.sum(A,axis=0)))

#Min and Max

import numpy
n=int(input().split()[0])
a=numpy.array([input().split() for i in range(n)],int)
 
print(numpy.max(numpy.min(a, axis=1)))   


#Mean, Var, and Std

import numpy


arr = []
N,M = map(int, input().split())
for _ in range(N):
     arr.append(list(map(int, input().split())))
arr = numpy.array(arr)

print(numpy.mean(arr, axis=1))
print(numpy.var(arr, axis=0))
print(round(numpy.std(arr), 11))


#Dot and Cross

import numpy

n=int(input())

A=numpy.array([list(map(int,input().split())) for _ in range(n)])

B=numpy.array([list(map(int,input().split())) for _ in range(n)])

print(numpy.dot(A,B))


#Inner and Outer

import numpy
A=input().split()
B=input().split()

A=numpy.array(A,int)
B=numpy.array(B,int)

print(numpy.inner(A,B))
print(numpy.outer(A,B))

#Polynomials

import numpy

A=numpy.array(input().split(),float)
x=float(input())

print(numpy.polyval(A,x))


#Linear Algebra

import numpy
N=int(input())
A=numpy.array([input().split() for _ in range(N)],float)
numpy.set_printoptions(legacy='1.13')

print(numpy.linalg.det(A))

##################################

#PRONLEM 2

#Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    count=0
    massimo=0
    for i in range(len(candles)):
        if candles[i]>=massimo:
            massimo=candles[i]
    for i in range(len(candles)):
        if candles[i]==massimo:
            count+=1
                
        
    return count
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

	
#Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if (x2-x1)*(v2-v1)<0 and (x2 - x1) % (v2 - v1) == 0:
        return'YES'
    else:
        return 'NO'
 
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

	
#Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    cum=0
    rec=5
    for i in range(n):
        rec = math.floor(rec/2)
        cum += rec
        rec *= 3
    return cum
        
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    a = str(sum([int(i) for i in n]))
    if len(n) == 1:
        if k == 1:
            return n
        else:
            return superDigit(n * k, 1)
    else:
        return superDigit(a, k)
    # Write your code here
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

	
#Insertion Sort - Part 1


import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    # Write your code here
    a = arr[-1]
    i=n-2
    while a <arr[i] and i >=0:
        arr[i+1]=arr[i]
        print(*arr)
        i-= 1

    arr[i+1] = a
    print(*arr)  

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    # Write your code here
    for i in range(1,n):
        a=arr[i]
        j=i-1
        while j>=0 and arr[j]>a:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=a
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)




	



	
	
