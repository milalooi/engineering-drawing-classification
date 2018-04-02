
# coding: utf-8

# In[1]:

import numpy as np
from PIL import Image
import os
import errno
from tqdm import tqdm


# In[3]:

N = 10
N2 = 100
NN = 20
THRESHOLD = 160
m = 4
n = 4

def getImageGreyScale(path):
    with Image.open(path) as img:
        w, h = img.size
        img = img.convert('L')
        w = int(w / 20) * 20
        h = int(h / 20) * 20
#         img.show()
        return np.mat(img.resize((w, h))), w, h

def getPixel(cell):
    if np.average(cell,axis=1).min() < THRESHOLD or np.average(cell,axis=0).min() < THRESHOLD:
        return 0
    return 255

def getCorner(cell, sx, sy):
    tab = np.zeros((N, N))
    for j in range(N):
        J = sy * j
        for i in range(N):
            I = sx * i
            tab[j][i] = getPixel(cell[J:J+sy, I:I+sx])
    return tab

def getCorners(img, w, h):

    scx = int(.5 + w / m)
    scy = int(.5 + h / n)
    sx = int(.5 + scx / N)
    sy = int(.5 + scy / N)

    a = getCorner(img[0:scy, 0:scx], sx, sy)
    b = getCorner(img[0:scy, scx*(m-1):], sx, sy)
    c = getCorner(img[scy*(n-1):, 0:scx], sx, sy)
    d = getCorner(img[scy*(n-1):, scx*(m-1):], sx, sy)
    return a, b, c, d

def compose(a, b, c, d):
    tab = []
    for j in range(NN):
        if j < N:
            tab.extend(a[j])
            tab.extend(b[j])
        else:
            tab.extend(c[j - N])
            tab.extend(d[j - N])
    img = Image.new('L', (NN, NN))
    img.putdata(tab)
    return img

def img2data(path):
    img, w, h = getImageGreyScale(path)
    a, b, c, d = getCorners(img, w, h)
    img = compose(a, b, c, d)
#     img.show()
    return np.asarray(img, dtype=np.uint8).flatten()/255


# In[4]:

def sigmoid(z):
    hx = 1./(1. + np.exp(-z))
    return hx


def backsigmoid(z,one):
    sig = sigmoid(z)
    return sig*(one - sig)


def J(y,h,output_size):
    cost = 0 
    for i in range (output_size):
        cost += -y[i]*np.log(h[i]) - (1.-y[i])*np.log(1.-h[i]) 
    return cost


def reg(theta1, theta2, m, n, hidden_size,output_size,lamb):
    _sum = 0
    for i in range (hidden_size):
        for j in range (1,n+1):
            _sum += pow(theta1[i,j],2)
    
    for i in range (output_size):
        for j in range (1,hidden_size+1):
            _sum += pow(theta2[i,j],2)
            
    cost2 = (lamb/(2*m))*_sum
    return cost2 


# In[5]:

def real_rotation(options, X, weight_array_path):
    output_size = options['output_size']
    hidden_size = options['hidden_size']
    sequence_size = 1
    alp = options['alp']
    lamb = options['lamb']
    
    
    #import data
    m,n = X.shape
    X_T = np.transpose(X)          # dimension (400,1)
    
    # import/generate weights
    theta1 = np.genfromtxt(weight_array_path[0], delimiter = ',')
    theta2 = np.genfromtxt(weight_array_path[1], delimiter = ',')

    inputshape1 = theta1.shape
    inputshape2 = theta2.shape
    if theta1.shape != (hidden_size,n+1):
        theta1 = np.transpose(theta1)
    if theta2.shape != (output_size,hidden_size+1):
        theta2 = np.transpose(theta2)
    if theta1.shape != (hidden_size,n+1) or theta2.shape != (output_size,hidden_size+1):
        print("bad theta shape")
        return

    
    #generate bias and zeros
    bias = np.ones((1,m))
    zero = np.zeros((1,m))
    one = np.ones((hidden_size+1,m))
                     
    # forward propagation
    Xnew = np.r_[bias,X_T]            
    z2 = np.matmul(theta1,Xnew)       
    a2 = sigmoid(z2)                  

    a2 = np.r_[bias,a2]               
    z3 = np.matmul(theta2,a2)         
    a3 = sigmoid(z3)                  
    a3_T = np.transpose(a3)           

    return a3_T


# In[ ]:


#### main script for rotation ####


##### locate the input and output folder ####
input_folder = r'D:\CENOZAI_AVEVA+GTD+ATTIT\Data\Others_all'
output_folder = r'D:\CENOZAI_AVEVA+GTD+ATTIT\Data\P&ID_Testing -OK'

#theta for rotation
theta1_rotation = 'theta1_rotation.txt'
theta2_rotation = 'theta2_rotation.txt'


# count number of files
list = os.listdir(input_folder) 
number_files=len(list)
print ("Number of files: %s" %number_files)


file_list = []
for root, dirs, files in os.walk(input_folder):
    for name in files:
        file_list.append(os.path.join(root,name))
print (file_list)


for f in tqdm(file_list):
    
    with Image.open(f) as img:
        width, height = img.size
   
        # check if file require rotation
        if width > height:
#             print('Landscape')
            split_path = os.path.split(f)
            f_name = split_path[1] 
            IMG = img.rotate(0, expand=True)
            IMG.save(output_folder +"/"+ f_name)
        else:
            # rotating process
            split_path = os.path.split(f)
            f_name = split_path[1] 
            f_data =  np.matrix(img2data(f))
            output = real_rotation(options, f_data, [theta1_rotation, theta2_rotation])
            needed_angle = np.argmax(output)*90
#             print("Anticlockwise", needed_angle)           
            IMG = img.rotate(needed_angle, expand=True)
            IMG.save(output_folder +"/"+ f_name)

