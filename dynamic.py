import numpy as np
import scipy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from scipy.spatial.distance import euclidean
import math

def plot_canvas(matrix, title, artist_coords = None):

    sns.set(style="whitegrid")
    plt.figure(figsize=(5,5))
    sns.heatmap(matrix, cmap='Blues', linewidths=0.025, cbar=False, square=True, annot=False, fmt='d', xticklabels=False, yticklabels=False, linecolor='black')

    if artist_coords is not None:
        coords_list = [tuple(coord) for coord_array in artist_coords for coord in coord_array if coord is not None]

        if coords_list:

            rows, cols = zip(*coords_list)
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)
            plt.gca().add_patch(plt.Rectangle((min_col-0.5, min_row-0.5), (max_col-min_col)+1, (max_row-min_row)+1, fill=None, edgecolor='red', lw=2))

    plt.title(title)
    plt.show()

def get_cnt_coords(n, k):
    if n % k != 0 or k % 2 == 0:
        raise ValueError("Invalid dimensions. k must be odd and be a factor of n.")
    
    coords = []
    
    for i in range(0, n, k):
        for j in range(0, n, k):
            subgrid_x_start = i
            subgrid_y_start = j
            subgrid_x_center = subgrid_x_start + (k - 1) // 2
            subgrid_y_center = subgrid_y_start + (k - 1) // 2
            coords.append((subgrid_x_center, subgrid_y_center))
    
    return coords

def get_cnt_coords(n, k, radius):
    if n % k != 0 or k % 2 == 0:
        raise ValueError("Invalid dimensions. k must be odd and be a factor of n.")
    
    coords = []
    
    for i in range(0, n, k):
        for j in range(0, n, k):
            subgrid_x_start = i
            subgrid_y_start = j
            # Adjust subgrid center for buffer
            subgrid_x_center = subgrid_x_start + (k - 1) // 2 + radius
            subgrid_y_center = subgrid_y_start + (k - 1) // 2 + radius
            coords.append((subgrid_x_center, subgrid_y_center))
    
    return coords


def find_matches(array1, array2):
    # Convert arrays to sets for faster lookup
    set1 = set(tuple(row) for row in array1)
    set2 = set(tuple(row) for row in array2)
    
    # Find the intersection (common elements) between the sets
    common_elements = set1.intersection(set2)
    
    return np.array([list(row) for row in common_elements])

def activate_cnts(artist_matrix, matches):
    for match in matches:
        artist_matrix[tuple(match)] = 1
    return artist_matrix

def reset_cnts(artist_matrix):
    return np.zeros_like(artist_matrix)

def draw_on_canvas(canvas, matches):
    for coord in matches:
        canvas[coord[0], coord[1]] = 1

    return canvas

def get_closest_pair(target_coords, cnt_coords):

    l2_norms = []
    for target_coord in target_coords:
        for cnt_coord in cnt_coords:
            l2_norms.append(euclidean(target_coord, cnt_coord))

    min_idx = np.argmin(l2_norms)
    target_idx = min_idx // len(cnt_coords)
    cnt_idx = min_idx % len(cnt_coords)
    
    target_coord = target_coords[target_idx]
    cnt_coord = cnt_coords[cnt_idx]

    return np.array(target_coord), np.array(cnt_coord)

def remove_matches(target_coords, matches):
    return np.array([coord for coord in target_coords if not np.any(np.all(coord == matches, axis=1))])


def determine_direction(target_coord, cnt_coord):

    target_x, target_y = target_coord
    cnt_x, cnt_y = cnt_coord

    if target_x == cnt_x and target_y == cnt_y:
        return "STAY"
    else:
        if target_x > cnt_x:
            return "UP"
        elif target_x < cnt_x:
            return "DOWN"
        elif target_y > cnt_y:
            return "LEFT"
        elif target_y < cnt_y:
            return "RIGHT"

def shift_canvas(canvas, direction):

    if direction == "STAY":
        return canvas
    rows, cols = canvas.shape
    shifted_canvas = np.zeros_like(canvas) 

    if direction == "UP":
        shifted_canvas[:-1, :] = canvas[1:, :]  
    elif direction == "DOWN":
        shifted_canvas[1:, :] = canvas[:-1, :] 
    elif direction == "LEFT":
        shifted_canvas[:, :-1] = canvas[:, 1:]  
    elif direction == "RIGHT":
        shifted_canvas[:, 1:] = canvas[:, :-1] 

    return shifted_canvas

def add_buffer(matrix, radius):
    buffered_dim = matrix.shape[0] + 2 * radius
    buffered_matrix = np.zeros((buffered_dim, buffered_dim), dtype=matrix.dtype)
    buffered_matrix[radius:-radius, radius:-radius] = matrix
    return buffered_matrix

canvas_dim = 15
sub_dim = 5
buffer = math.floor(sub_dim/2)
buffer = sub_dim*2
total_dim = canvas_dim + buffer

# make all
source = np.round(np.random.rand(canvas_dim, canvas_dim)).astype(int)
canvas = np.zeros((canvas_dim,canvas_dim)).astype(int)
artist = np.zeros((canvas_dim, canvas_dim)).astype(int)

source = add_buffer(source, buffer)
canvas = add_buffer(canvas, buffer)
artist = add_buffer(artist, buffer)
# were adding more buffers to artist, but its fine for now

# canvas = np.zeros((total_dim,total_dim)).astype(int)
# artist = np.zeros((total_dim, total_dim)).astype(int)

target_coords = np.argwhere(source == 1)
cnt_coords = get_cnt_coords(canvas_dim, sub_dim, buffer)

plot_canvas(source, 'source')
time.sleep(2)

# while target coords exist
while len(target_coords) > 0:

    target_coord, cnt_coord = get_closest_pair(target_coords, cnt_coords)
    direction = determine_direction(target_coord, cnt_coord)
    canvas = shift_canvas(canvas, direction)

    if direction == "UP":
        target_coords[:, 0] -= 1
    elif direction == "DOWN":
        target_coords[:, 0] += 1
    elif direction == "LEFT":
        target_coords[:, 1] -= 1
    elif direction == "RIGHT":
        target_coords[:, 1] += 1
        
    matches = find_matches(target_coords, cnt_coords)
    
    if len(matches) > 0:
        artist = activate_cnts(artist, matches)
        canvas = draw_on_canvas(canvas, matches)
        artist = reset_cnts(artist)
        target_coords = remove_matches(target_coords, matches)

    clear_output(wait=True)
    plot_canvas(canvas, 'drawing...')
    # time.sleep(1)

clear_output(wait=True)
plot_canvas(canvas, 'done')
plot_canvas(source, 'source')

# centered_final_canvas = canvas[buffer:-buffer, buffer:-buffer]
# clear_output(wait=True)
# plot_canvas(centered_final_canvas, '.')