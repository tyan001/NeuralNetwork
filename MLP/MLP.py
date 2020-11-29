#!/usr/bin/env python
# coding: utf-8

# **Thony Yan <br>**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import copy


# # Multilayer Perceptron (MLP)
# 

# In[2]:


p1 = np.array([-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,-1,1])
p6 = np.array([1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,1,1])
p11 = np.array([-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1])
p16 = np.array([1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1])
p21 = np.array([1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1])


# In[3]:


p1 = p1.reshape(4,5)
p6 = p6.reshape(4,5)
p11 = p11.reshape(4,5)
p16 = p16.reshape(4,5)
p21 = p21.reshape(4,5)


# In[4]:


p1.shape


# In[5]:


base = [p1,p6,p11,p16,p21]


# In[6]:


def show(figs): # This function is use to show image
    img=plt.figure(figsize=(10, 10))
    for i in range(len(figs)):
        img.add_subplot(5, 5, i+1)
        plt.imshow(figs[i]-1, cmap='Greys')


# In[7]:


img=plt.figure(figsize=(8, 8))
for i in range(1,6):
    img.add_subplot(5, 5, i)
    plt.imshow(base[i-1]-1, cmap='Greys')


# In[8]:


a_mod = np.array([[2,1],[3,2],[3,4],[4,4]])
e_mod = np.array([[2,2],[4,2],[5,3],[4,3]])
i_mod = np.array([[1,1],[4,2],[1,4],[4,3]])
o_mod = np.array([[2,2],[4,2],[2,3],[4,3]])
u_mod = np.array([[2,1],[4,1],[2,3],[4,3]])


# In[9]:


mod = [a_mod,e_mod,i_mod,o_mod,u_mod]


# In[10]:


def modify(base, mod):
    lst = []
    lst.append(base)
    for i in range(len(mod)):
        tmp = copy.deepcopy(base)
        tmp[mod[i][1]-1,mod[i][0]-1] *= -1
        lst.append(tmp)

    return lst


# In[11]:


a_test = modify(base[0], mod[0])
e_test = modify(base[1], mod[1])
i_test = modify(base[2], mod[2])
o_test = modify(base[3], mod[3])
u_test = modify(base[4], mod[4])


# In[12]:


show(a_test)
show(e_test)
show(i_test)
show(o_test)
show(u_test)


# In[13]:


test_set = np.array([a_test,e_test,i_test,o_test,u_test])


# In[14]:


tset_mod1 = np.array([[ [4,1],[5,3],[2,4],[1,2] ],  #tset1 a
                      [ [2,4],[5,2],[4,3],[1,3] ],  #tset1 e
                      [ [4,4],[2,4],[2,2],[3,4] ],  #tset1 i
                      [ [1,3],[3,2],[5,4],[3,3] ],  #tset1 o
                      [ [5,3],[3,4],[1,4],[5,4] ]]) #tset1 u


# In[15]:


tset1_a = modify(base[0], tset_mod1[0])
tset1_e = modify(base[1], tset_mod1[1])
tset1_i = modify(base[2], tset_mod1[2])
tset1_o = modify(base[3], tset_mod1[3])
tset1_u = modify(base[4], tset_mod1[4])


# In[16]:


show(tset1_a)
show(tset1_e)
show(tset1_i)
show(tset1_o)
show(tset1_u)


# In[17]:


tset1 = np.array([tset1_a,tset1_e,tset1_i,tset1_o,tset1_u])


# In[18]:


tset_mod2 = np.array([[ [3,2],[1,3],[3,3],[2,1] ], #tset2 a
                      [ [3,2],[3,3],[5,4],[1,2] ], #tset2 e
                      [ [1,3],[2,2],[4,3],[1,4] ], #tset2 i
                      [ [4,3],[4,3],[5,1],[2,4] ], #tset2 o
                      [ [3,3],[2,2],[2,1],[1,3] ]])#tset2 u


# In[19]:


def modify2(base,mod):
    lst = []
    lst.append(base[0])
    for i in range(1,5):
        tmp = copy.deepcopy(base[i])
        tmp[mod[i-1][1]-1,mod[i-1][0]-1] *= -1
        lst.append(tmp)
    return lst


# In[20]:


tset2_a = modify2(tset1_a,tset_mod2[0])
tset2_e = modify2(tset1_e,tset_mod2[1])
tset2_i = modify2(tset1_i,tset_mod2[2])
tset2_o = modify2(tset1_o,tset_mod2[3])
tset2_u = modify2(tset1_u,tset_mod2[4])


# In[21]:


show(tset2_a)
show(tset2_e)
show(tset2_i)
show(tset2_o)
show(tset2_u)


# In[22]:


tset2 = np.array([tset2_a,tset2_e,tset2_i,tset2_o,tset2_u])


# In[23]:


tset_mod3 = np.array([[ [5,2],[3,3],[1,3],[5,1] ], #tset3 a
                      [ [3,4],[1,4],[5,2],[1,3] ], #tset3 e
                      [ [1,1],[1,2],[2,4],[2,1] ], #tset3 i
                      [ [1,2],[4,2],[3,3],[1,4] ], #tset3 o
                      [ [3,2],[5,2],[5,4],[4,2] ]])#tset3 u


# In[24]:


tset3_a = modify2(tset2_a,tset_mod3[0])
tset3_e = modify2(tset2_e,tset_mod3[1])
tset3_i = modify2(tset2_i,tset_mod3[2])
tset3_o = modify2(tset2_o,tset_mod3[3])
tset3_u = modify2(tset2_u,tset_mod3[4])


# In[25]:


show(tset3_a)
show(tset3_e)
show(tset3_i)
show(tset3_o)
show(tset3_u)


# In[26]:


tset3 = np.array([tset3_a,tset3_e,tset3_i,tset3_o,tset3_u])


# In[27]:


class Neural_Network:

    def __init__(self):
        self.weights = [] # weight matrices
        self.bias = [] # bias matrices
        self.activation = [] # activation functions
        self.z_val = [np.zeros(1)] # sum values of neurons. note: first value suppose to be inputs
        self.a_val = [np.zeros(1)] # values after activation functions are apply. note: first value suppose to be inputs
        self.sensitivity = [] # sensitivity or delta (derivatives)


    def add_layer(self, neurons: int, activation: str, input_shape=None):
        if input_shape is None:
            try:
                w = np.random.uniform(0,0.25,(self.weights[-1].shape[1],neurons))
                self.weights.append(w)
            except IndexError:
                w = np.random.uniform(0,0.25,(1,neurons))
                self.weights.append(w)

        else:
            input = np.prod(input_shape)
            w = np.random.uniform(0,0.25,(input,neurons))
            self.weights.append(w)

        self.bias.append(np.random.rand(neurons))
        self.activation.append(self.activation_f(activation))
        self.z_val.append(np.zeros(neurons))
        self.a_val.append(np.zeros(neurons))


    @staticmethod
    def sigmoid(x, derivative: bool=False):
        z = 1/(1+np.exp(-x))
        if derivative:
            z = z * (1-z)
        return z

    @staticmethod
    def tanh(x, derivative: bool=False):
        z =  (1-np.exp(-2*x)) / (1+np.exp(-2*x))
        if derivative:
            z = (1 + z) * (1 - z)
        return z

    @staticmethod
    def linear(x, derivative: bool=False):
        if derivative:
            return np.array(1)
        return x

    def activation_f(self, activation_name: str):

        activation = {
            'linear' : self.linear,
            'sigmoid' : self.sigmoid,
            'tanh' : self.tanh
        }

        act = str.lower(activation_name)

        if act in activation:
            return activation[act]
        else:
            print("activation function not in record")

    @staticmethod
    def mse(error):

        mse = np.sum(error ** 2)
        return mse

    @staticmethod
    def error(target, output):
        error = (target - output)
        return error
    
    @staticmethod
    def hardlim(output, tresh=0.35):
        
        output[output>tresh] = 1
        output[output<tresh] = -1
        return output
    
    @staticmethod
    def max_arg(output):
        
        index = output.argmax()
        output.fill(-1)
        output[index] = 1
        return output
        
    def feedforward(self, x):

        self.a_val[0] = x
        self.z_val[0] = x

        for layer in range(len(self.weights)):

            z = np.dot(self.a_val[layer], self.weights[layer]) + self.bias[layer]
            a = self.activation[layer](z)
            self.z_val[layer+1] = z
            self.a_val[layer+1] = a

    def back_propagate(self, error):
        error = -2 * error # -2 * (target-output)
        self.sensitivity = []
        for i in reversed(range(len(self.weights))):

            s = error * self.activation[i](self.z_val[i+1], derivative=True) # -2 * FMnM * (t-a)
            self.sensitivity.insert(0,s)
            error = np.dot(s, self.weights[i].T)

    def update_weights(self, alpha=0.1):
        for i in range(len(self.weights)):
            sensitivity = self.sensitivity[i]
            a = self.a_val[i]

            sensitivity = sensitivity.reshape(sensitivity.shape[0],-1)
            a = a.reshape(a.shape[0],-1)

            sa = np.dot(sensitivity,a.T)

            self.weights[i] = self.weights[i] - (alpha * sa.T)
            self.bias[i] = self.bias[i] - (alpha * self.sensitivity[i])

    def train(self, x, y, alpha=0.01):

        self.feedforward(x)
        error = self.error(y, self.a_val[-1])
        self.back_propagate(error)
        self.update_weights(alpha)
        return self.mse(error)

    def predict(self,x):
        self.a_val[0] = x
        self.z_val[0] = x
        for layer in range(len(self.weights)):

            z = np.dot(self.a_val[layer], self.weights[layer]) + self.bias[layer]
            a = self.activation[layer](z)
            self.z_val[layer+1] = z
            self.a_val[layer+1] = a
        
        print(self.max_arg(a))


# In[28]:


y = np.array([1.,-1.,-1.,-1.,-1.])

inputs = np.array(a_test[0].flatten(), dtype=np.float)
inputs


# In[29]:


model = Neural_Network()


# In[30]:



model.add_layer(10, 'sigmoid', input_shape=(4,5))
model.add_layer(5, 'tanh')


# In[31]:


print(model.weights[0].shape)
print(model.weights[1].shape)
print(model.bias[0].shape)
print(model.bias[1].shape)


# In[32]:


model.feedforward(inputs)


# In[33]:


model.a_val


# In[34]:


model.z_val


# In[35]:


error = model.error(y, model.a_val[-1])
error


# In[36]:


s = model.activation[1](model.z_val[2], derivative=True) * error
st = s.reshape(s.shape[0],-1)
st


# In[37]:


a = model.a_val[1]
a = a.reshape(a.shape[0],-1)
a


# In[38]:


Wnew = np.dot(st,a.T)


# In[39]:


Wnew.T.shape


# In[40]:


model.weights[1].shape


# In[41]:


model.weights[1]


# In[42]:


model.back_propagate(error)


# In[43]:


model.sensitivity[1].shape


# In[44]:


model.bias[1].shape


# In[45]:


model.weights[0][14]


# In[46]:


model.update_weights()


# In[47]:


print(model.weights[0].shape)
print(model.weights[1].shape)
print(model.bias[0].shape)
print(model.bias[1].shape)


# In[48]:


a = model.a_val
t = np.dot(inputs, model.weights[0])
t


# In[49]:


model = Neural_Network()
model.add_layer(10, 'sigmoid', input_shape=(4,5))
model.add_layer(5, 'tanh')


# In[50]:


training_set = a_test, e_test, i_test, o_test, u_test
training_set


# In[51]:


y_set = np.array([[1.,-1,-1,-1,-1], [-1,1,-1,-1,-1],[-1,-1,1,-1,-1],[-1,-1,-1,1,-1],[-1,-1,-1,-1,1]])


# In[52]:


epoch = 100
total_mse = []
random_w1 = []
random_w2 = []
random_w3 = []
bias1 = []
bias2 = []
random1 = np.random.randint(20)
random2 = np.random.randint(10)
random3 = np.random.randint(10)
random4 = np.random.randint(5)
bias_random1 = np.random.randint(10)
bias_random2 = np.random.randint(5)
for i in range(epoch):
    mse = 0
    random_w1.append(model.weights[0][random1][random2])
    random_w2.append(model.weights[1][random3][random4])
    bias1.append(model.bias[0][bias_random1])
    bias2.append(model.bias[1][bias_random2])

    for i in range(len(training_set)):
        set = training_set[i]
        target = y_set[i]
        for j in range(len(set)):
            #show(set)
            #print(target)
            mse += model.train(set[i].flatten(), target, 0.1)

    mse = (mse / 25.)
    total_mse.append(mse)
    


# In[53]:


for i in range(len(training_set)):
    set = training_set[i]
    for j in range(len(set)):
        model.predict(set[i].flatten())

    print('')


# In[54]:


plt.plot(total_mse)
plt.xlabel('epoch')
plt.ylabel('mse value')
plt.title('mse')


# In[55]:


total_mse


# In[56]:


plt.plot(random_w1)
plt.xlabel('epoch')
plt.ylabel('weight value')
plt.title('layer 1, weight (' + str(random1) + ',' + str(random2) + ')')


# In[57]:


plt.plot(random_w2)
plt.xlabel('epoch')
plt.ylabel('weight value')
plt.title('layer 2, weight (' + str(random3) + ',' + str(random4) + ')')


# In[58]:


plt.plot(bias1)
plt.xlabel('epoch')
plt.ylabel('bias value')
plt.title('layer 1, bias (' + str(bias_random1) + ')')


# In[59]:


plt.plot(bias2)
plt.xlabel('epoch')
plt.ylabel('bias value')
plt.title('layer 2, bias (' + str(bias_random2) + ')')


# In[60]:


def fit(model,epochs, alpha, dataset, targets, exit=0.01, input_l=20, hidden_l=10, output_l=5):
    total_mse = []
    random_w1 = []
    random_w2 = []
    random_w3 = []
    bias1 = []
    bias2 = []
    random1 = np.random.randint(input_l)
    random2 = np.random.randint(hidden_l)
    random3 = np.random.randint(hidden_l)
    random4 = np.random.randint(output_l)
    random5 = np.random.randint(input_l)
    random6 = np.random.randint(hidden_l)
    bias_random1 = np.random.randint(hidden_l)
    bias_random2 = np.random.randint(output_l)
    for epoch in range(epochs):
        mse = 0
        random_w1.append(model.weights[0][random1][random2])
        random_w2.append(model.weights[1][random3][random4])
        random_w3.append(model.weights[0][random5][random6])
        bias1.append(model.bias[0][bias_random1])
        bias2.append(model.bias[1][bias_random2])

        for i in range(len(dataset)):
            sets = dataset[i]
            target = targets[i]
            for j in range(len(sets)):
                mse += model.train(sets[i].flatten(), target, 0.1)

        mse = (mse / 25.)
        total_mse.append(mse)
        if mse < exit:
            break
    
    plt.plot(total_mse)
    plt.xlabel('epoch')
    plt.ylabel('mse value')
    plt.title('mse')
    plt.show()
    
    plt.plot(random_w1)
    plt.xlabel('epoch')
    plt.ylabel('weight value')
    plt.title('layer 1, weight (' + str(random1) + ',' + str(random2) + ')')
    plt.show()
    
    plt.plot(random_w2)
    plt.xlabel('epoch')
    plt.ylabel('weight value')
    plt.title('layer 2, weight (' + str(random3) + ',' + str(random4) + ')')
    plt.show()
    
    plt.plot(random_w3)
    plt.xlabel('epoch')
    plt.ylabel('weight value')
    plt.title('layer 1, weight (' + str(random5) + ',' + str(random6) + ')')
    plt.show()
    
    plt.plot(bias1)
    plt.xlabel('epoch')
    plt.ylabel('bias value')
    plt.title('layer 1, bias (' + str(bias_random1) + ')')
    plt.show()
    
    plt.plot(bias2)
    plt.xlabel('epoch')
    plt.ylabel('bias value')
    plt.title('layer 2, bias (' + str(bias_random2) + ')')
    plt.show()
    
    print('\n number of epochs to reach an mse of ' , epoch)


# In[61]:


model1 = Neural_Network()
model1.add_layer(10, 'sigmoid', input_shape=(4,5))
model1.add_layer(5, 'tanh')


# In[62]:


fit(model1,epochs=100, alpha=0.1, dataset=training_set, targets=y_set, exit=0.005)


# In[63]:


model2 = Neural_Network()
model2.add_layer(10, 'sigmoid', input_shape=(4,5))
model2.add_layer(5, 'tanh')


# In[64]:


fit(model2,epochs=100, alpha=0.01, dataset=training_set, targets=y_set, exit=0.005)


# In[65]:


model3 = Neural_Network()
model3.add_layer(10, 'sigmoid', input_shape=(4,5))
model3.add_layer(5, 'tanh')


# In[66]:


fit(model3,epochs=100, alpha=0.001, dataset=training_set, targets=y_set, exit=0.005)


# In[67]:


for i in range(len(tset1)):
    sets = tset1[i]
    for j in range(len(sets)):
        model1.predict(sets[i].flatten())

    print('')


# In[68]:


for i in range(len(tset2)):
    sets = tset2[i]
    for j in range(len(sets)):
        model1.predict(sets[i].flatten())

    print('')


# In[69]:


for i in range(len(tset3)):
    sets = tset3[i]
    for j in range(len(sets)):
        model1.predict(sets[i].flatten())

    print('')

