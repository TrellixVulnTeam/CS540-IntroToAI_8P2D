#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import math
import string
import numpy as np

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)


def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict.fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        #1: Store case-folded file text in a new variable "newText"
        newText = f.read().upper()
        #print(newText)
        
        #2: Update frequency of each letter in the dictionary
        for letter in newText:
            if letter in string.ascii_uppercase:
                if letter in X:
                    X[letter]+=1
                else:
                    X[letter] = 1
    return X


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!


# In[2]:


#Q1: Building Digital Shredder
print("Q1")
newDict = shred("samples/letter1.txt")
#newDict = shred("letter.txt")
for key, value in newDict.items():
    print(key,value)


# In[3]:


#Q2: Computes X1 log e1 and X1 log s1.

#~~~Provided in write-up~~~
#Prior Probability that the letter is of English
ePrior = 0.6
#Provided prior probability that the letter is of Spanish
sPrior = 0.4

#Access tuple of count vectors (characters probabilities of English and Spanish)
paramVectors = get_parameter_vectors()
#print(paramVectors)

#~~~Computation~~~
dictVals = newDict.values()
countList = list(dictVals)

#Create a vector of the character counts
countVector = np.array(countList)

#Computation #1: A_1 * log(e_1)
eNum = countVector[0] * math.log(paramVectors[0][0])
eNum = round(eNum, 4)

#Computation #2: A_1 * log(s_1)
sNum = countVector[0] * math.log(paramVectors[1][0])
sNum = round(sNum, 4)

#Print Values
print("Q2")
print(format(eNum, ".4f"))
print(format(sNum, ".4f"))


# In[4]:


#Q3: Computes F(English) and F(Spanish)
#Compute F(English) and F(Spanish). Similarly, print
#“Q3” followed by their values up to 4 decimal places on two separate lines.
#p is e if y = English or s if y = Spanish

#~~~Computation~~~
#Computation #1: F(English)
f_english = math.log(ePrior) + sum(countVector[i]*math.log(paramVectors[0][i]) for i in range (0, 26))
f_english = round(f_english, 4)

#Computation #1: F(Spanish)
f_spanish = math.log(sPrior) + sum(countVector[i]*math.log(paramVectors[1][i]) for i in range (0, 26))
f_spanish = round(f_spanish, 4)

#Print Values
print("Q3")
print(format(f_english, ".4f"))
print(format(f_spanish, ".4f"))


# In[5]:


#Q4: Computes P(Y = English | X)
#~~~Suggested if-else check per the specification~~~
if (f_spanish - f_english) >= 100:
    prob_e = 0
if (f_spanish - f_english) <= -100:
    prob_e = 1
else:
    prob_e = 1/(1 + math.exp(f_spanish - f_english))
prob_e = round(prob_e, 4)

#Print Values
print("Q4")
print(format(prob_e, ".4f"))

