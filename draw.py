import tkinter as tk
from PIL import Image
import os
import random
import string


def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y), width=15)
    lasx, lasy = event.x, event.y

def save(event):
    # save postscipt image
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    canvas.postscript(file = name + '.eps')
    # use PIL to convert to PNG 
    img = Image.open(name + '.eps')
    for f in os.listdir('./myfig/'):
        os.remove(os.path.join('./myfig/', f))
    resized_img = img.resize((28, 28), Image.ANTIALIAS)
    resized_img.save('./myfig/' + name + '.png', 'png')
    os.remove(name + '.eps')

root = tk.Tk()
root.geometry("800x600")

canvas = tk.Canvas(root)
canvas.pack(anchor='nw', fill='both', expand=1)
canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)
root.bind("<Control-s>", save)

root.mainloop()
