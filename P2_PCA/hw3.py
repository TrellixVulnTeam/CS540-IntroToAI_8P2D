#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# load the dataset from a provided .npy file, re-center it around the origin, 
# and return it as a numpy array of floats.
def load_and_center_dataset(filename):
    # Your implementation goes here!
    # 1.1 Load the Dataset
    x = np.load(filename) # n × d dataset -> numPy array
    # 1.2 Center the Dataset around the origin
    return x - np.mean(x, axis = 0) # array of means for each data point x_mean[2414]

# calculate and return the covariance matrix of the dataset as a numpy
# matrix (d × d array -> 1024 x 1024).
def get_covariance(dataset):
    # Your implementation goes here!
    cov_matrix = []

    # 1.1 Find the transpose of X
    trans_x = np.transpose(dataset)
    
    # 1.2 Compute Covariance Matrix
    cov_matrix = (1/(len(dataset)-1)) * np.dot(trans_x, dataset)
        
    return cov_matrix

# perform eigendecomposition on the covariance matrix S and return a diagonal matrix
# (numpy array) with the largest m eigenvalues on the diagonal in descending order, and a matrix (numpy
# array) with the corresponding eigenvectors as columns.
def get_eig(S, m):
    # Your implementation goes here!
    # ~~ 1 Call eigh() library function on Covariance Matrix ~~
    n = len(S)
    eig_val, eig_vec = eigh(S, subset_by_index= [n-m, n-1])
    
    # ~~ 2 Create a diagonal array of m largest eigenvalues ~~
   
    # 2.1 Sort the eigen values in descending order
    sorted_eig_val = -np.sort(-eig_val)
    
    # 2.2 Put the eigenvalue sorted array in a diagonal matrix
    eig_val = np.diag(sorted_eig_val)

    # ~~ 3 Keep the eigenvectors in the corresponding columns after sorting eigenvalues ~~
    eig_vec[:, [1,0]] = eig_vec[:, [0,1]] # Switches 1st and 2nd columns of the eigenvector matrix
    
    return eig_val, eig_vec

# similar to get_eig, but instead of returning the first m, return all eigenvalues
# and corresponding eigenvectors in a similar format that explain more than a prop proportion
# of the variance (specifically, please make sure the eigenvalues are returned in descending order).
def get_eig_prop(S, prop):
    # Your implementation goes here!
    # ~~ 1 Call eigh() library function on Covariance Matrix, for all values and vectors ~~
    eig_val, eig_vec = eigh(S)
    
    # ~~ 2 Create a diagonal array of eigenvalues ~~
    eig_val = np.diag(eig_val)
    
    # ~~ 3 Call eigh() for values that explain more than certain proportion of variance ~~
    eig_val, eig_vec = eigh(S, subset_by_value=[(np.trace(eig_val)*prop), np.inf] )
    
    # ~~ 4 Sort the eigen values in descending order, and put in a diagonal matrix ~~
    eig_val = -np.sort(-eig_val)
    eig_val = np.diag(eig_val)

     # ~~ 5 Keep the eigenvectors in the corresponding columns after sorting eigenvalues ~~
    eig_vec[:, [1,0]] = eig_vec[:, [0,1]] # Switches 1st and 2nd columns of the eigenvector matrix
   
    return eig_val, eig_vec

# project each d × 1 image into your m-dimensional subspace (spanned by
# m vectors of size d × 1) and return the new representation as a d × 1 numpy array.
def project_image(image, U):
    # Your implementation goes here!
    sum = 0
    for col in range(len(U[0])):
        alpha = np.dot((U[:, col].T), image)
        sum += np.dot(alpha, U[:, col])
    return sum

# use matplotlib to display a visual representation of the original image
# and the projected image side-by-side.
def display_image(orig, proj):
    # Your implementation goes here!
    # ~~ 1 Reshape the images to be 32 × 32 and transpose them to make them straight ~~
    
    orig_image = np.reshape(orig, (32, 32))
    orig_image = orig_image.T
    proj_image = np.reshape(proj, (32, 32))
    proj_image = proj_image.T
    
    # ~~ 2 Create a figure with one row of two subplots ~~
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # ~~ 3 Give titles to the subplots ~~
    ax1.set_title("Original")
    ax2.set_title("Projection")
    
    # ~~ 4 Use 'imshow' with optional argument-> aspect ='equal' ~~
    
    colorBar1 = ax1.imshow(orig_image, aspect='equal')
    colorBar2 = ax2.imshow(proj_image, aspect='equal')
    
     # ~~ 5 Create a colorbar for each image ~~
    fig.colorbar(colorBar1, ax= ax1)
    fig.colorbar(colorBar2, ax= ax2)
    
    # ~~ 6 Render the plots ~~
    plt.show()
    return

