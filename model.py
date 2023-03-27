from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

class GEN(object):
	def __init__(self):
		self.model = self.create_model()
	
	def res_conv_block(self,filters,num,tensor):
		x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), name='conv' + str(num) + 'a',padding='same', use_bias=False)(tensor)
		x = layers.BatchNormalization(axis=3)(x)
		x = layers.LeakyReLU(alpha=0.1)(x)
		x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), name='conv' + str(num) + 'c',padding='same', use_bias=False)(x)
		x = layers.BatchNormalization(axis=3)(x)
		shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), name='conv' + str(num) + 'b',padding='same', use_bias=False)(tensor)
		shortcut = layers.BatchNormalization(axis=3)(shortcut)
		res_path = layers.add([shortcut, x])
		res_path = layers.LeakyReLU(alpha=0.1)(res_path)    #Activation after addition with shortcut (Original residual block)
		return res_path

	
	def repeat_elem(self,tensor, rep):
    	# lambda function to repeat Repeats the elements of a tensor along an axis
    	#by a factor of rep.
    	# If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    	#(None, 256,256,6), if specified axis=3 and rep=2.

		return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),arguments={'repnum': rep})(tensor)


	def gating_signal(self,filters,num,tensor):
		x = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), name='gating_num' + str(num),padding='same', use_bias=False)(tensor)
		x = layers.BatchNormalization(axis=3)(x)
		x = layers.LeakyReLU(alpha=0.1)(x)
		return x

	
	def attention_block(self,tensor,gating,filters):
		shape_tensor = K.int_shape(tensor)
		shape_gating = K.int_shape(gating)
    		
		# Getting the x signal to the same shape as the gating signal
		theta_tensor = layers.Conv2D(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(tensor)  
		shape_theta_tensor = K.int_shape(theta_tensor)

    	# Getting the gating signal to the same number of filters as the inter_shape
		phi_gating = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(gating)
		upsample_gating = layers.Conv2DTranspose(filters, kernel_size=(3, 3),strides=(shape_theta_tensor[1] // shape_gating[1], shape_theta_tensor[2] // shape_gating[2]),padding='same')(phi_gating)  

		concat_tengat = layers.add([upsample_gating, theta_tensor])
		act_tengat = layers.LeakyReLU(alpha=0.1)(concat_tengat)
		psi = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(act_tengat)
		sigmoid_tengat = layers.Activation('sigmoid')(psi)
		shape_sigmoid = K.int_shape(sigmoid_tengat)
		upsample_psi = layers.UpSampling2D(size=(shape_tensor[1] // shape_sigmoid[1], shape_tensor[2] // shape_sigmoid[2]))(sigmoid_tengat)  

		upsample_psi = self.repeat_elem(upsample_psi, shape_tensor[3])

		y = layers.multiply([upsample_psi, tensor])

		result = layers.Conv2D(shape_tensor[3], kernel_size=(1, 1), padding='same')(y)
		result_bn = layers.BatchNormalization()(result)
		return result_bn


	

	def create_model(self):
		Input = layers.Input(shape=(256,256,3))
		

		# Downsampling layers
		layer1 = self.res_conv_block(32,1,Input)
		pool1 = layers.MaxPooling2D(pool_size=(2,2))(layer1)
        		
		layer2 = self.res_conv_block(64,2,pool1)
		pool2 = layers.MaxPooling2D(pool_size=(2,2))(layer2)

		layer3 = self.res_conv_block(128,3,pool2)
		pool3 = layers.MaxPooling2D(pool_size=(2,2))(layer3)

		layer4 = self.res_conv_block(256,4,pool3)
		pool4 = layers.MaxPooling2D(pool_size=(2,2))(layer4)

		layer5 = self.res_conv_block(512,5,pool4)
		pool5 = layers.MaxPooling2D(pool_size=(2,2))(layer5)

		layer6 = self.res_conv_block(1024,6,pool5)



		# Upsampling layers
		gating1 = self.gating_signal(512,1,layer6)
		att1 = self.attention_block(layer5, gating1,512)
		up_layer1 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(layer6)
		up_layer1 = layers.concatenate([up_layer1, att1], axis=3)
		layer7 = self.res_conv_block(512,7,up_layer1)
			

		gating2 = self.gating_signal(256,2,layer7)
		att2 = self.attention_block(layer4, gating2,256)
		up_layer2 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(layer7)
		up_layer2 = layers.concatenate([up_layer2, att2], axis=3)
		layer8 = self.res_conv_block(256,8,up_layer2)


		gating3 = self.gating_signal(128,3,layer8)
		att3 = self.attention_block(layer3, gating3,128)
		up_layer3 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(layer8)
		up_layer3 = layers.concatenate([up_layer3, att3], axis=3)
		layer9 = self.res_conv_block(128,9,up_layer3)


		gating4 = self.gating_signal(64,4,layer9)
		att4 = self.attention_block(layer2, gating4,64)
		up_layer4 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(layer9)
		up_layer4 = layers.concatenate([up_layer4, att4], axis=3)
		layer10 = self.res_conv_block(64,10,up_layer4)


		gating5 = self.gating_signal(32,5,layer10)
		att5 = self.attention_block(layer1, gating5,32)
		up_layer5 = layers.UpSampling2D(size=(2,2), data_format="channels_last")(layer10)
		up_layer5 = layers.concatenate([up_layer5, att5], axis=3)
		layer11 = self.res_conv_block(32,11,up_layer5)


 		
		layer12 = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(layer11)
		layer12 = layers.BatchNormalization(axis=3)(layer12)
		layer12 = layers.LeakyReLU(alpha=0.1)(layer12)
		layer12 = layers.MaxPooling2D(pool_size=(2,2))(layer12)
		
		Output = layers.Dense(2,activation='sigmoid',name='Output')(layer12) 
		model = models.Model(Input,Output)
		return model

