
import cv2
import numpy as np
import os
import random

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def contrasting_color(color):
    return (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)

def draw_circle(img, colorful=True):
    center_x = random.randint(10, 54)
    center_y = random.randint(10, 54)
    radius = random.randint(5, 30)
    color = random_color() if colorful else contrasting_color(img[0,0])
    cv2.circle(img, (center_x, center_y), radius, color, -1)
    return img

def draw_square(img, colorful=True):
    start_x = random.randint(2, 54)
    start_y = random.randint(2, 54)
    side_length = random.randint(5, 30)
    color = random_color() if colorful else contrasting_color(img[0,0])
    cv2.rectangle(img, (start_x, start_y), (start_x + side_length, start_y + side_length), color, -1)
    return img

def draw_triangle(img, colorful=True):
    center_x = random.randint(10, 54)
    center_y = random.randint(10, 54)
    length = random.randint(10, 30)
    angle = random.random() * 2 * np.pi
    point1 = (center_x + int(length * np.cos(angle)), center_y + int(length * np.sin(angle)))
    point2 = (center_x + int(length * np.cos(angle + 2*np.pi/3)), center_y + int(length * np.sin(angle + 2*np.pi/3)))
    point3 = (center_x + int(length * np.cos(angle + 4*np.pi/3)), center_y + int(length * np.sin(angle + 4*np.pi/3)))
    color = random_color() if colorful else contrasting_color(img[0,0])
    pts = np.array([point1, point2, point3], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    cv2.fillPoly(img, [pts], color)
    return img

shapes = [draw_circle, draw_square, draw_triangle]
shape_names = ['circle', 'square', 'triangle']

for idx, shape in enumerate(shapes):
    for i in range(1000):  # Generate 1000 images per shape
        colorful = True if i >= 500 else False
        bg_color = random.choice([(0, 0, 0), (255, 255, 255)]) if i % 2 == 0 else random_color()
        img = np.ones((64, 64, 3), dtype=np.uint8) * bg_color
        img = shape(img, colorful)
        directory = f'data/{shape_names[idx]}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(f'{directory}/{i}.png', img)
