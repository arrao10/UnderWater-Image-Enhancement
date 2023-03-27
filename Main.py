import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import sqlite3
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import os
import training as tr
import prediction as pre

bgcolor="#ADD8E6"
bgcolor1="#FF7F50"
fgcolor="black"

def Home():
	global window
	
	def clear():
		print("Clear1")
		txt1.delete(0, 'end')    

	window = tk.Tk()
	window.title("Underwater Image Enchancement")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="UNDERWATER IMAGE ENHANCEMENT" ,bg=bgcolor  ,fg=fgcolor  ,width=75  ,height=3,font=('times', 20, 'bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Training Folder",width=18  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=90, y=200)
	
	txt = tk.Entry(window,width=30,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=340, y=215)


	lbl1 = tk.Label(window, text="Select Image",width=18  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl1.place(x=90, y=300)
	
	txt1 = tk.Entry(window,width=30,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt1.place(x=340, y=315)

	#lbl4 = tk.Label(window, text="Notification : ",width=20  ,fg=fgcolor,bg=bgcolor  ,height=2 ,font=('times', 15, ' bold underline ')) 
	#lbl4.place(x=100, y=400)

	#message = tk.Label(window, text="" ,bg="white"  ,fg="black",width=60  ,height=8, activebackground = bgcolor ,font=('times', 15, ' bold ')) 
	#message.place(x=400, y=400)

	def browse():
		path=filedialog.askdirectory(initialdir=os.getcwd(),title='Please Select a Folder')
		print(path)
		txt.delete('0',"end")
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input Error", "Select Training Folder")	

	def browse1():
		path=filedialog.askopenfilename()
		print(path)
		txt1.delete('0',"end")
		txt1.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select Image")	

	def train():
		path=txt.get()
		if path !="":
			print(path)
			tr.process(path)
			tm.showinfo("Input", "Training Finished Successfully")	
		else:
			tm.showinfo("Input error", "Select Training Folder")	

	def predict():
		path=txt1.get()
		if path !="":
			pre.process(path)
			print(path)
			tm.showinfo("Input", "Prediction Finished Successfully")	
		else:
			tm.showinfo("Input Error", "Select Image")	
			

	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=680, y=200)
	
	browse1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse1.place(x=680, y=300)

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
	clearButton.place(x=970, y=300)
	 

	trainbutton = tk.Button(window, text="Train", command=train  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	trainbutton.place(x=620, y=600)

	prebutton = tk.Button(window, text="Enchance", command=predict  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	prebutton.place(x=820, y=600)

	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1020, y=600)

	window.mainloop()
Home()

