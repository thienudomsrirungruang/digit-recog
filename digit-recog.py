import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

import os

import numpy as np
import scipy.ndimage.filters

import tensorflow as tf
from tensorflow import keras

import threading
import time

import tkinter

# Objects

class ImageState():
    def __init__(self, image):
        self.image = image
        self.changed_state = True

class Probabilities():
    def __init__(self, probabilities):
        self.probabilities = probabilities

class ProgramStatus():
    def __init__(self):
        self.running = True

# Threads

class DigitDrawer(threading.Thread):
    def __init__(self, program_status, image_state, scale=20):
        threading.Thread.__init__(self)
        self.program_status = program_status
        self.image_state = image_state
        self.scale = scale
    
    def init_window(self):
        self.screen = pygame.display.set_mode((28 * self.scale, 28 * self.scale))
        self.draw_image()

    def draw_image(self):
        self.screen.fill((255, 255, 255))
        for i in range(28):
            for j in range(28):
                self.screen.fill([255 - int(255 * self.image_state.image[j, i]) for k in range(3)],
                                    pygame.Rect(i * self.scale, j * self.scale, self.scale, self.scale))
        pygame.display.flip()

    def clear_image(self):
        self.image_state.image = np.zeros((28, 28))
        self.image_state.changed_state = True

    # Wu's algorithm: draws lines with antialiasing
    # code from https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
    # returns iterable of tuple ((x, y), strength)
    def get_line(self, coords1, coords2):
        x0 = coords1[0]
        y0 = coords1[1]
        x1 = coords2[0]
        y1 = coords2[1]
        steep = (y1 - y0) > abs(x1 - x0)

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = y1 - y0
        gradient = 1.0
        if dx != 0:
            gradient = dy/dx
        
        # first endpoint
        xend = int(x0+0.5) # round
        yend = y0 + gradient * (xend - x0)
        xgap = 1 - (x0+0.5)%1
        xpxl1 = xend
        ypxl1 = int(yend)
        if steep:
            yield((ypxl1, xpxl1), (1-yend%1) * xgap)
            yield((ypxl1+1, xpxl1), yend%1 * xgap)
        else:
            yield((xpxl1, ypxl1), (1-yend%1) * xgap)
            yield((xpxl1, ypxl1+1), yend%1 * xgap)
        intery = yend + gradient

        # second endpoint
        xend = int(x1+0.5) # round
        yend = y1 + gradient * (xend - x1)
        xgap = (x1+0.5) % 2
        xpxl2 = xend
        ypxl2 = int(yend)
        if steep:
            yield((ypxl2, xpxl2), (1-yend%1) * xgap)
            yield((ypxl2+1, xpxl2), yend%1 * xgap)
        else:
            yield((xpxl2, ypxl2), (1-yend%1) * xgap)
            yield((xpxl2, ypxl2), yend%1 * xgap)
        
        # main loop
        if steep:
            for x in range(xpxl1+1, xpxl2):
                yield((int(intery), x), 1-intery%1)
                yield((int(intery)+1, x), intery%1)
                intery = intery + gradient
        else:
            for x in range(xpxl1+1, xpxl2):
                yield((x, int(intery)), 1-intery%1)
                yield((x, int(intery)+1), intery%1)
                intery = intery + gradient

    # strength between 0 and 1.0
    def draw_point(self, point, strength=1.0, brush_sigma=.6, brush_magnitude=5):
        x, y = np.array(point).astype(int)
        new_point_array = np.zeros((28,28))
        new_point_array[y, x] = brush_magnitude * strength
        new_point_array = scipy.ndimage.filters.gaussian_filter(new_point_array, sigma=brush_sigma) # make the brush "fat"
        combined_array = np.clip(self.image_state.image + new_point_array, 0, 1) # limit values between 0 and 1
        self.image_state.image = combined_array
        self.image_state.changed_state = True

    def draw_line(self, coords1, coords2):
        coords1 = tuple((np.array(coords1) / self.scale))
        coords2 = tuple((np.array(coords2) / self.scale))
        for point in self.get_line(coords1, coords2):
            if 0 <= point[0][0] <= 27 and 0 <= point[0][1] <= 27:
                self.draw_point(*point)

    def run(self):
        print("DigitDrawer started")
        self.init_window()
        drawing = False
        previous_drawing = False
        previous_pos = None
        while self.program_status.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.program_status.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                    self.draw_point(tuple(np.array(pygame.mouse.get_pos()) / self.scale))
                if event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                    previous_drawing = False
                if event.type == pygame.MOUSEMOTION:
                    if drawing and previous_drawing:
                        self.draw_line(previous_pos, pygame.mouse.get_pos())
                    previous_drawing = drawing
                    previous_pos = pygame.mouse.get_pos()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.clear_image()
            self.draw_image()
    
class DigitRecognizer(threading.Thread):
    def __init__(self, program_status, image_state, probabilities):
        threading.Thread.__init__(self)
        self.program_status = program_status
        self.image_state = image_state
        self.probabilities = probabilities
        self.checkpoint_path = "training_conv/cp-{epoch:04d}.ckpt".format(epoch=6)
    
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.load_weights(self.checkpoint_path)
        return model

    def predict(self, image):
        image = (np.reshape(image, (1, 28, 28, 1)))
        return self.model.predict(image)[0]

    def run(self):
        print("DigitRecognizer started")
        print("Creating model...")
        self.model = self.create_model()
        while self.program_status.running:
            while not self.image_state.changed_state and self.program_status.running:
                time.sleep(0.1)
            image = self.image_state.image
            prediction = self.predict(image)
            self.probabilities.probabilities = prediction
            self.image_state.changed_state = False

def init_graph(window, probabilities):

    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    canvas = FigureCanvasTkAgg(fig, master=window)
    # canvas.show()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    line = ax.bar(np.arange(0, 10), probabilities.probabilities[0])

    ax.set_xticks(np.arange(0, 10))
    ax.set_ylim(0, 1)

    canvas.draw()
    return ax, canvas

def update_graph(ax, canvas, window, probabilities):

    ax.clear()
    prediction = np.where(probabilities.probabilities == np.amax(probabilities.probabilities))[0][0]
    line = ax.bar(np.arange(0, 10), probabilities.probabilities, color=["red" if prediction == i else "blue" for i in range(10)])
    
    ax.set_xticks(np.arange(0, 10))
    ax.set_ylim(0, 1)

    ax.set_title("{0:.1f}% sure it's a {1:1d}".format(np.max(probabilities.probabilities) * 100, prediction))
    
    canvas.draw()
    window.after(100, update_graph, ax, canvas, window, probabilities)

def graph(probabilities, window):
    ax, canvas = init_graph(window, probabilities)
    update_graph(ax, canvas, window, probabilities)
    window.mainloop()

if __name__ == '__main__':


    window = tkinter.Tk()

    program_status = ProgramStatus()

    # random starting image from dataset:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    image_state = ImageState((train_images[np.random.randint(0, 10000)] / 255).reshape(28, 28))

    # blank starting image:
    # image_state = ImageState(np.zeros((28, 28)))
    probabilities = Probabilities(np.zeros(10))
    digit_drawer_thread = DigitDrawer(program_status, image_state, 20)
    digit_recognizer_thread = DigitRecognizer(program_status, image_state, probabilities)
    digit_drawer_thread.start()
    digit_recognizer_thread.start()

    graph(probabilities, window)

    
