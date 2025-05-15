import math
import random
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import proj3d
import heapq
import os

# Function to calculate 3D Euclidean distance
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

# A* algorithm for pathfinding
def a_star(start, goal, obstacles, grid_size):
    def heuristic(p1, p2):
        return calculate_distance(p1, p2)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    grid = [[[0 for _ in range(grid_size)] for _ in range(grid_size)] for _ in range(grid_size)]
    for obstacle in obstacles:
        x, y, z = obstacle[1]
        grid[x][y][z] = 1        # Mark obstacle as 1

    open_set = [(0, start)]      # Priority queue: (f_score, position)
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current is not None:        # Backtrack to find the path
                path.append(current)
                current = came_from[current]
            return path[::-1]

        x, y, z = current
        for direction in directions:
            nx, ny, nz = x + direction[0], y + direction[1], z + direction[2]
            neighbor = (nx, ny, nz)
            if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                if grid[nx][ny][nz] == 1:
                    continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

# Function to update logo position dynamically
# Function to update logo position dynamically
def update_logo(ax, submarine_position, ab):
    # Get the 2D projection of the 3D submarine position
    x2d, y2d, _ = proj3d.proj_transform(submarine_position[0], submarine_position[1], submarine_position[2], ax.get_proj())
    # Update the AnnotationBbox offset
    ab.xybox = (x2d, y2d)
    # Redraw only the artist to minimize lag
    ab.figure.canvas.draw_idle()

# Generate the plot and show in Tkinter
def show_plot():
    grid_size = 50  # Smaller grid for faster testing
    submarine_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))
    goal_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))

    # Ensure goal is distinct from submarine
    while goal_position == submarine_position:
        goal_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))

    num_obstacles = 20
    obstacles = []
    for i in range(num_obstacles):
        pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        while pos == submarine_position or pos == goal_position:
            pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        obstacles.append((f"Mine_{i+1}", pos))

    obstacle_distances = [(obstacle[0], obstacle[1], calculate_distance(submarine_position, obstacle[1])) for obstacle in obstacles]
    obstacle_distances.sort(key=lambda x: x[2])

    path = a_star(submarine_position, goal_position, obstacles, grid_size)

    if not path:
        messagebox.showinfo("No Path", "No valid path found to the goal.")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles using mine.png (real naval mine photo)
    mine_path = "mine.png"  # Path to real mine image
    if os.path.exists(mine_path):
        try:
            mine_img = mpimg.imread(mine_path)
            for obstacle in obstacle_distances:
                x, y, z = obstacle[1]
                imagebox = OffsetImage(mine_img, zoom=0.015)  #! Further reduced zoom for real mine photo (not working)
                x2d, y2d, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
                ab_mine = AnnotationBbox(imagebox, (x, y, z), xybox=(x2d, y2d), frameon=False, xycoords='data', boxcoords="data", pad=0.0)
                ax.add_artist(ab_mine)
                ax.text(x + 1, y + 1, z + 1, obstacle[0], fontsize=8)
        except Exception as e:
            print(f"Error loading mine.png: {e}")
            # Fallback to red dots if mine.png fails
            for obstacle in obstacle_distances:
                ax.scatter(obstacle[1][0], obstacle[1][1], obstacle[1][2], c='r', marker='o')
                ax.text(obstacle[1][0] + 1, obstacle[1][1] + 1, obstacle[1][2] + 1, obstacle[0], fontsize=8)
    else:
        print(f"Mine image file {mine_path} not found. Using default red dots.")
        # Fallback to red dots if mine.png is missing
        for obstacle in obstacle_distances:
            ax.scatter(obstacle[1][0], obstacle[1][1], obstacle[1][2], c='r', marker='o')
            ax.text(obstacle[1][0] + 1, obstacle[1][1] + 1, obstacle[1][2] + 1, obstacle[0], fontsize=8)

    # Plot submarine (invisible point to anchor logo)
    ax.plot([submarine_position[0]], [submarine_position[1]], [submarine_position[2]], marker='o', markersize=0.1, alpha=0)

    # Load and display submarine logo
    logo_path = "submarine_logo.png"
    ab = None
    if os.path.exists(logo_path):
        try:
            sub_logo = mpimg.imread(logo_path)
            imagebox = OffsetImage(sub_logo, zoom=0.1)
            x2d, y2d, _ = proj3d.proj_transform(submarine_position[0], submarine_position[1], submarine_position[2], ax.get_proj())
            ab = AnnotationBbox(imagebox, (x2d, y2d), frameon=False, xycoords='data')
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error loading logo: {e}")
    else:
        print(f"Logo file {logo_path} not found. Using default marker.")
        ax.scatter(submarine_position[0], submarine_position[1], submarine_position[2], c='b', marker='o', s=100)

    # Update logo position on view change
    if ab is not None:
        def on_view_change(event):
            update_logo(ax, submarine_position, ab)

        # Connect to both draw_event and motion_notify_event for robust updates
        fig.canvas.mpl_connect('draw_event', on_view_change)
        fig.canvas.mpl_connect('motion_notify_event', on_view_change)

    ax.text(submarine_position[0] + 1, submarine_position[1] + 1, submarine_position[2] + 1, "Submarine", fontsize=10)
    ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='y', marker='^', s=150, label='Goal')

    path_x, path_y, path_z = zip(*path)
    ax.plot(path_x, path_y, path_z, c='g', label='Path')

    ax.set_title('Submarine Pathfinding in 3D')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    ax.grid(True)

    plt.show()
    messagebox.showinfo("Path Found", f"Path from submarine to goal found with {len(path)} steps.")

# Run the simulation
show_plot()
