from model import GEN
import cv2
import numpy as np

def process(path):
	model = GEN()
	model.model.load_weights('tr.h5')
	image = cv2.imread(path)
	image = cv2.resize(image,(256,256))
	image = np.divide(image,255.0)
	image = np.expand_dims(image,axis=0)
	pred = model.model.predict(image)
	alpha = pred[0][0][0][0]
	beta = pred[0][0][0][1]
	print(alpha,beta)
	image = cv2.imread(path)
	image = cv2.resize(image,(512,512))
	pred = cv2.convertScaleAbs(image,alpha = alpha,beta = beta*50)
	cv2.imshow('Result Image',pred)
	cv2.imshow('Original Image',image)
	cv2.waitKey(0)
	cv2.imwrite('Result Image.jpg',pred)
