import numpy as np

def calculate_distance(focal_length, known_width, width_in_frame):
    return (known_width * focal_length) / width_in_frame

def find_distances(focal_length, known_width, detections):
    distances = []
    for detection in detections:
        width = detection[2]
        distances.append(calculate_distance(focal_length, known_width, width))
    return distances

def get_distances(faces):
    KNOWN_WIDTH = 11.0  
    focal_length = 500  
    distances = find_distances(focal_length, KNOWN_WIDTH, faces)
    return distances