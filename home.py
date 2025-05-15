
import math
import random
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import proj3d
import heapq
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        grid[x][y][z] = 1  # Mark obstacle as 1

    open_set = [(0, start)]  # Priority queue: (f_score, position)
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current is not None:
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
def update_logo(ax, submarine_position, ab):
    x2d, y2d, _ = proj3d.proj_transform(submarine_position[0], submarine_position[1], submarine_position[2], ax.get_proj())
    ab.xybox = (x2d, y2d)
    ab.figure.canvas.draw_idle()

# Main application class for enhanced front end
class SubmarinePathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Submarine Pathfinding Simulation")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title
        ttk.Label(self.main_frame, text="Submarine Pathfinding in 3D", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        # Input frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Simulation Parameters", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # Grid size input
        ttk.Label(input_frame, text="Grid Size (10-100):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.grid_size_entry = ttk.Entry(input_frame, width=10)
        self.grid_size_entry.insert(0, "50")
        self.grid_size_entry.grid(row=0, column=1, sticky=tk.W)

        # Number of obstacles input
        ttk.Label(input_frame, text="Number of Obstacles (1-50):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.obstacles_entry = ttk.Entry(input_frame, width=10)
        self.obstacles_entry.insert(0, "20")
        self.obstacles_entry.grid(row=1, column=1, sticky=tk.W)

        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot).grid(row=0, column=1, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready to run simulation")
        ttk.Label(self.main_frame, textvariable=self.status_var, foreground="blue").grid(row=3, column=0, columnspan=2, pady=5)

        # Plot frame
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.rowconfigure(4, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # Initialize plot
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title('Submarine Pathfinding in 3D')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_zlabel('Z Coordinate')
        self.ax.grid(True)
        self.canvas.draw()
        self.status_var.set("Plot cleared")

    def run_simulation(self):
        try:
            grid_size = int(self.grid_size_entry.get())
            num_obstacles = int(self.obstacles_entry.get())
            if grid_size < 10 or grid_size > 100:
                messagebox.showerror("Error", "Grid size must be between 10 and 100.")
                return
            if num_obstacles < 1 or num_obstacles > 50:
                messagebox.showerror("Error", "Number of obstacles must be between 1 and 50.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
            return

        self.status_var.set("Running simulation...")
        self.root.update()

        # Generate random positions
        submarine_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        goal_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))

        while goal_position == submarine_position:
            goal_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1), random.randint(0, grid_size-1))

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
            self.status_var.set("No valid path found")
            messagebox.showinfo("No Path", "No valid path found to the goal.")
            return

        # Clear previous plot
        self.ax.clear()

        # Plot obstacles using mine.png
        mine_path = "mine.png"
        if os.path.exists(mine_path):
            try:
                mine_img = mpimg.imread(mine_path)
                for obstacle in obstacle_distances:
                    x, y, z = obstacle[1]
                    imagebox = OffsetImage(mine_img, zoom=0.015, alpha=0.7)  # Added transparency
                    x2d, y2d, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
                    ab_mine = AnnotationBbox(imagebox, (x, y, z), xybox=(x2d, y2d), frameon=False, xycoords='data', boxcoords="data", pad=0.0)
                    self.ax.add_artist(ab_mine)
                    self.ax.text(x + 1, y + 1, z + 1, obstacle[0], fontsize=8, color='darkred')
            except Exception as e:
                print(f"Error loading mine.png: {e}")
                for obstacle in obstacle_distances:
                    self.ax.scatter(obstacle[1][0], obstacle[1][1], obstacle[1][2], c='r', marker='o', alpha=0.7)
                    self.ax.text(obstacle[1][0] + 1, obstacle[1][1] + 1, obstacle[1][2] + 1, obstacle[0], fontsize=8, color='darkred')
        else:
            print(f"Mine image file {mine_path} not found. Using default red dots.")
            for obstacle in obstacle_distances:
                self.ax.scatter(obstacle[1][0], obstacle[1][1], obstacle[1][2], c='r', marker='o', alpha=0.7)
                self.ax.text(obstacle[1][0] + 1, obstacle[1][1] + 1, obstacle[1][2] + 1, obstacle[0], fontsize=8, color='darkred')

        # Plot submarine
        self.ax.plot([submarine_position[0]], [submarine_position[1]], [submarine_position[2]], marker='o', markersize=0.1, alpha=0)

        # Load and display submarine logo
        logo_path = "submarine_logo.png"
        ab = None
        if os.path.exists(logo_path):
            try:
                sub_logo = mpimg.imread(logo_path)
                imagebox = OffsetImage(sub_logo, zoom=0.1, alpha=0.8)
                x2d, y2d, _ = proj3d.proj_transform(submarine_position[0], submarine_position[1], submarine_position[2], self.ax.get_proj())
                ab = AnnotationBbox(imagebox, (x2d, y2d), frameon=False, xycoords='data')
                self.ax.add_artist(ab)
            except Exception as e:
                print(f"Error loading logo: {e}")
        else:
            print(f"Logo file {logo_path} not found. Using default marker.")
            self.ax.scatter(submarine_position[0], submarine_position[1], submarine_position[2], c='b', marker='o', s=100, alpha=0.8)

        # Update logo position on view change
        if ab is not None:
            def on_view_change(event):
                update_logo(self.ax, submarine_position, ab)
            self.canvas.mpl_connect('draw_event', on_view_change)
            self.canvas.mpl_connect('motion_notify_event', on_view_change)

        self.ax.text(submarine_position[0] + 1, submarine_position[1] + 1, submarine_position[2] + 1, "Submarine", fontsize=10, color='navy')
        self.ax.scatter(goal_position[0], goal_position[1], goal_position[2], c='gold', marker='^', s=150, label='Goal', alpha=0.9)

        path_x, path_y, path_z = zip(*path)
        self.ax.plot(path_x, path_y, path_z, c='limegreen', label='Path', linewidth=2)

        # Enhance plot styling
        self.ax.set_title('Submarine Pathfinding in 3D', fontsize=14, pad=20, color='darkblue')
        self.ax.set_xlabel('X Coordinate', fontsize=10, color='darkblue')
        self.ax.set_ylabel('Y Coordinate', fontsize=10, color='darkblue')
        self.ax.set_zlabel('Z Coordinate', fontsize=10, color='darkblue')
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_facecolor('#e6f3ff')  # Light blue background
        self.fig.patch.set_facecolor('#e6f3ff')

        self.canvas.draw()
        self.status_var.set(f"Path found with {len(path)} steps")
        messagebox.showinfo("Path Found", f"Path from submarine to goal found with {len(path)} steps.")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SubmarinePathfindingApp(root)
    root.mainloop()
    root.destroy()  


    