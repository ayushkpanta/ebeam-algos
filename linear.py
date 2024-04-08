# np.random.seed(1232)
import numpy as np # type: ignore
import scipy
import os

# generating a random binary matrix for testing
canvas_dim = 100
source = np.round(np.random.rand(canvas_dim, canvas_dim)).astype(int)
target_coords = np.argwhere(source == 1)

# also have this for manual testing
# source_matrix = np.array([[0,0,0,0,0,0],
#                    [0,0,0,0,0,0],
#                    [0,0,0,0,0,0],
#                    [0,0,0,0,0,0],
#                    [0,0,0,0,0,0]])

canvas = np.zeros((canvas_dim,canvas_dim)).astype(int)

artist_dim = 20
artist = np.zeros((artist_dim, artist_dim)).astype(int)

artist_coords = np.zeros((artist_dim, artist_dim, 2), dtype=int)
for i in range(artist_dim):
    for j in range(artist_dim):
        artist_coords[i, j] = [i, j]


# show a matriximport numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

def plot_canvas(matrix, title, artist_coords = None):

    sns.set(style="whitegrid")
    plt.figure(figsize=(10,10))
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

# # translates a matrix in proper direction
def translate(direction, position_array, artist_dim):

    # copy just cuz
    new_position_array = np.copy(position_array)
    

    if direction == "U":
        new_position_array[:, :, 0] -= artist_dim
    elif direction == "D":
        new_position_array[:, :, 0] += artist_dim
    elif direction == "L":
        new_position_array[:, :, 1] -= artist_dim
    elif direction == "R":
        new_position_array[:, :, 1] += artist_dim

    return new_position_array

def wall_check_and_change(direction, position_array, canvas, artist_dim):
    curr_x_idx = position_array[1, 1, 1]
    max_index = canvas.shape[1]
    min_index = 0
    
    if direction == "R" and curr_x_idx + artist_dim >= max_index:
        direction = "D"
    elif direction == "L" and curr_x_idx - artist_dim < min_index:
        direction = "D"
    elif direction == "D":
        if curr_x_idx + artist_dim >= max_index:
            direction = "L"
        elif curr_x_idx - artist_dim < min_index:
            direction = "R"

    return direction

def draw(artist, artist_coords, canvas, source, target_coords, canvas_dim, artist_dim):

    canvas_x_lim, canvas_y_lim = canvas_dim, canvas_dim
    
    artist_x_max, artist_y_max = artist_coords[-1, -1, 0], artist_coords[-1, -1, 1]
    
    t = 1
    direction = "R"
    
    plot_canvas(source, f"Target Canvas", artist_coords)

    time.sleep(1)
    
    start = time.time()
    
    while not np.array_equal(canvas, source) and (artist_x_max < canvas_x_lim or artist_y_max < canvas_y_lim):

    
        selected_coords = np.array([]).reshape(0, 2).astype(int)
    
        for x in range(artist_dim):
            for y in range(artist_dim):
                artist_coord = artist_coords[x][y]
                matches = np.all(target_coords == artist_coord, axis=1)
                if np.any(matches):
                    selected_coords = np.vstack([selected_coords, artist_coord])
                    target_coords = target_coords[~matches]
    

        # wack with my generator
        for coord in selected_coords:
            index = np.argwhere(np.all(artist_coords == coord, axis=2))[0]
            artist[index[0], index[1]] = 1
            canvas[int(coord[0]), int(coord[1])] = 1
            artist[index[0], index[1]] = 0

        clear_output(wait=True)
        plot_canvas(canvas, f"Canvas at t = {t}", artist_coords)
    
        direction = wall_check_and_change(direction, artist_coords, canvas, artist_dim)
        artist_coords = translate(direction, artist_coords, artist_dim)
    
        artist_x_max, artist_y_max = artist_coords[-1, -1, 0], artist_coords[-1, -1, 1]
        
        t += 1
    
    end = time.time()
    
    print(f"Completed in {round(end - start, 3)} seconds for {canvas_dim}x{canvas_dim} canvas using {artist_dim}x{artist_dim} artist")
    
    print(f"Full match: {np.all(source == canvas)}")
    print(f"Pct: {np.sum(source == canvas) / np.prod(source.shape) * 100}%")
    
draw(artist, artist_coords, canvas, source, target_coords, canvas_dim, artist_dim)