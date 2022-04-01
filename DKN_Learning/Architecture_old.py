# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:27:38 2019

@author: dykua

Network architecture
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM,Conv2D, Input, Lambda, TimeDistributed, Conv1D, Flatten,Concatenate, Reshape, Add,BatchNormalization,Activation,Layer,MaxPooling2D,UpSampling2D,ConvLSTM2D,Conv3D,Conv3DTranspose
#from keras import regularizers
import tensorflow as tf
from tensorflow.keras.regularizers import l1,l2 
from tensorflow.keras.initializers import RandomNormal, Zeros,RandomUniform
import numpy as np
import keras.backend as K

def _transformer(x, out_dim, inter_dim_list=[32, 32],l2_reg=0, activation_out = 'tanh'):
    for j in inter_dim_list:
        x = (Dense(j,kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))(x)
        x = Activation('relu')(x)
    x = (Dense(out_dim, activation=activation_out,kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))(x)
    return x
    
def _pred_K(Gx, num_complex, num_real,hidden_widths_omega, K_reg,l2_reg=0,activation_out='linear'):
    x = TimeDistributed(Dense(hidden_widths_omega,kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))(Gx) 
    Koop = TimeDistributed(Dense(num_complex*2 + num_real, activity_regularizer = l1(K_reg), activation=activation_out, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-3),kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))(x)
    return Koop
    
class linear_update(Layer):

    def __init__(self, output_dim, num_complex, num_real, **kwargs):
        self.output_dim = output_dim
        self.kernels = []
        self.num_complex = num_complex
        self.num_real = num_real
        
        super(linear_update, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernels': self.kernels,
            'num_complex': self.num_complex,
            'num_real': self.num_real,
        })
        return config

    def build(self, input_shape):
#        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        super(linear_update, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        # Identify latent encoding and Koopman eigenvalues
        y, Km = x
        # Intialise list for the complex sequence of timesteps
        C_seq_i = [] 
        

        if self.num_complex:
            # For each timestep, use Gx and corresponding Koopman eigenvalue to compute the next timestep KGx           
            for i in range(self.output_dim[0]):
                y_i = y[:,i] # current latent encoding at time i
                Km_i = Km[:,i] # corresponding Koopman eigenvalue
                C_seq_c = [] # the current timestep complex components
                
                # for each complex pair
                for count_c in range(self.num_complex):
                
                    # forming complex block : batchsize, 2, 2
                    scale = tf.exp(Km_i[:, count_c])
                    cs = tf.cos(Km_i[:, count_c + self.num_complex])
                    sn = tf.sin(Km_i[:, count_c + self.num_complex])
                    real = tf.multiply(scale, cs)
                    img = tf.multiply(scale, sn)
                    block = tf.stack([real, -img, img, real], axis = 1)
                    Ci = tf.reshape(block, (-1, 2, 2))
                    
                    # Compute next timestep via eigenvalues 
                    y_i_c = y_i[:,(2*count_c):(2*count_c+2)] # Part of latent encoding corresponding to current complex pair
                    C_seq_c.append(tf.einsum('ik,ikj->ij', y_i_c, Ci)) 
                     
                # Create a tensor of all the components for this timestep
                C_seq_c_tensor = tf.reshape(tf.stack(C_seq_c, axis=2),(-1, 2*self.num_complex))                
                C_seq_i.append(C_seq_c_tensor)
  
            # Put all timesteps into a single tensor
            C_seq_tensor = tf.reshape(tf.stack(C_seq_i, axis = 1), (-1, self.output_dim[0], 2*self.num_complex))
       
        # forming real block: batchsize, 1
        R_seq = []
        if self.num_real:
            error('NOT IMPLEMENTED YET, STILL USES OLD AUTONOMOUS METHOD (NOT CONTROL)')
            R = tf.exp(Km[:,(2*self.num_complex):])
            R_seq.append(y[:,0, (2*self.num_complex):]) # previous version
            #R_seq.append(y[:, (2*self.num_complex):]) # Linear update only given the initial condition 
            for i in range(self.output_dim[0]-1):
                R_seq.append(tf.multiply(R_seq[i], R))
            R_seq_tensor = tf.stack(R_seq, axis = 1)

        if self.num_complex and self.num_real:
            return tf.concat([C_seq_tensor, R_seq_tensor], axis=2)
        
        elif self.num_real:
            return R_seq_tensor
        elif self.num_complex:
            return C_seq_tensor
 
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_y, shape_Km = input_shape
        return (shape_y[0], self.output_dim[0], self.output_dim[1])    
