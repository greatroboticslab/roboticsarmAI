

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, simpledialog
from time import sleep
import math

import numpy as np
import math
from dobot_util import Dobot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, simpledialog
from time import sleep
import math
from dobot_util import Dobot

import threading

# Ported directly from HongboRobot_ActualRobot_AI_Points.m
DRAWING_POINTS = np.array([
    [230, -30], [240, -30], [255, -30], [270, -30], [285, -30], [300, -30], [315, -30], [330, -30], [345, -30], [360, -30],
    [360, -20], [360, -5], [360, 10], [360, 25], [360, 40], [360, 55], [360, 70], [360, 85],
    [355, 90], [350, 95], [345, 100], [348, 105], [352, 110], [350, 115], [345, 120], [340, 125],
    [338, 130], [340, 135], [345, 140], [348, 145],
    [350, 140], [352, 130], [352, 115], [352, 95], [352, 75], [352, 55], [352, 35], [352, 15], [352, -5],
    [340, -5], [340, 10], [340, 30], [340, 50], [340, 70], [340, 90], [338, 105], [336, 115], [332, 120],
    [330, 118], [328, 110], [326, 98], [326, 80], [326, 60], [326, 40], [326, 20],
    [320, 20], [315, 22], [310, 25], [305, 32], [302, 40], [300, 52], [298, 40], [295, 32], [290, 25],
    [285, 22], [280, 20],
    [275, 20], [274, 35], [273, 50], [272, 70], [270, 90], [268, 110], [266, 120],
    [264, 110], [262, 95], [262, 70], [262, 45], [262, 20],
    [260, 20], [250, 20], [240, 20],
    [240, -5], [240, 15], [240, 35], [240, 55], [240, 75], [240, 95], [240, 115],
    [238, 125], [235, 130], [232, 135], [235, 140], [238, 145],
    [240, 140], [242, 130], [244, 115], [244, 95], [244, 75], [244, 55], [244, 35], [244, 15], [244, -5],
    [230, -5], [230, 10], [230, 30], [230, 50], [230, 70], [230, 85],
    [235, 90], [240, 95], [245, 100], [250, 105], [248, 110], [244, 115], [242, 120], [240, 125],
    [238, 130], [240, 135], [245, 140], [248, 145],
    [250, 140], [252, 130], [254, 115], [254, 95], [254, 75], [254, 55], [254, 35], [254, 15], [254, -5],
    [260, -5], [275, -5], [290, -5], [300, -5], [310, -5], [325, -5], [340, -5],
    [300, 10], [295, 15], [290, 22], [288, 32], [290, 42], [295, 50], [300, 55],
    [305, 50], [310, 42], [312, 32], [310, 22], [305, 15], [300, 10],
    [290, 60], [285, 65], [280, 70], [278, 80], [280, 90], [285, 98], [290, 102],
    [295, 104], [300, 105], [305, 104], [310, 102], [315, 98], [320, 90], [322, 80], [320, 70],
    [315, 65], [310, 60], [305, 58], [300, 57], [295, 58], [290, 60],
    [265, 30], [275, 40], [285, 50], [300, 65], [315, 50], [325, 40], [335, 30],
    [360, -30], [300, -30], [230, -30]
])


# --- NEW: Global Robot State ---
robot_data = {
    "joints": [0.0, 0.0, 0.0, 0.0],
    "cartesian": [0.0, 0.0, 0.0, 0.0],
    
}
is_jogging = False

def feedback_loop(robot_inst):
    """Thread function to constantly read Port 30004."""
    while True:
        try:
            data = robot_inst.feedback.get_feedback()
            if data is not None:
                # Store data in our global dictionary
                robot_data["joints"] = data[0]['q_actual'][:4].tolist()
                robot_data["cartesian"] = data[0]['tool_vector_actual'][:4].tolist()
        except:
            pass
        sleep(0.02) # 50Hz frequency

# Add this line where you initialize your robot connection:


class RobotManager:
    def __init__(self, ip="192.168.1.6", urdf_path=None):
        self.ip = ip
        self.robot = Dobot(self.ip, urdf_file=urdf_path) #

    def boot_robot(self):
        """Dashboard handshake for physical robot initialization"""
        print("here")
        self.robot.dashboard.ClearError()
        self.robot.dashboard.EnableRobot()
        print("Robot motors enabled.")

    def run_drawing(self):
        self.boot_robot()
        for i, pt in enumerate(DRAWING_POINTS):
            # Penal height management: lift pen for move to start, lower for drawing
            z_height = 245.0 if i == 0 or i == (len(DRAWING_POINTS) - 1) else 220.0
            
            # Use original Ikinematics function
            joints = Ikinematics(pt[0], pt[1], z=z_height) 
            
            # Send movement command to robot
            self.robot.movement.joint_mov_j(joints)
            print(f"Drawing point {i+1}/{len(DRAWING_POINTS)}: {pt}")
            
        self.robot.movement.sync()
        print("Drawing complete.")


import numpy as np
import math
from time import sleep

# Global variables for state tracking
robot = None
ROBOT_CONNECTED = False

def initialize_robot(ip="192.168.1.6"):
    global robot, ROBOT_CONNECTED

    try:
        from dobot_util import Dobot
        print(f"Attempting to connect to robot at {ip}...")
        
        # 1. Establish Network Connection
        robot = Dobot(ip, logging=True)
        
        # 2. Bootup Handshake (Critical for Physical Robot)
        # Clear existing alarms and power on the motors
        print("here2 ")
        robot.dashboard.send_command("ClearError()")
        sleep(0.5)
        print("Here3")
        robot.dashboard.enable()
        sleep(3)
        
        ROBOT_CONNECTED = True
        print("Robot connected and enabled successfully!")
        threading.Thread(target=feedback_loop, args=(robot,), daemon=True).start()
        robot.dashboard.send_command("CoordinateL(0)")
        robot.dashboard.set_user(0)
        robot.dashboard.set_tool(0)
        return True

    except Exception as e:
        print(f"Robot connection failed: {e}")
        print("Running in demo mode - robot commands will be simulated")
        robot = None
        ROBOT_CONNECTED = False
        return False

# Call the function immediately to maintain original behavior
initialize_robot("192.168.1.6")

# Calculate inverse kinematics for a 2-link planar arm
# --- CONFIGURATION TOGGLE ---
# False = Original Way (Checks X and Y values directly)
# True  = Newer Way (Checks calculated J1/J2 angles against degree limits)
STRICT_JOINT_CHECKING = True 

# CONFIGURATION TOGGLE
# Set to True to allow coordinates like 300 or 400 by checking angles instead of mm
STRICT_JOINT_CHECKING = True 

def Ikinematics(x, y, z=200.0, r=0.0):
    L1 = 200.0  # Length of first arm segment
    L2 = 200.0  # Length of second arm segment

    # Physical Joint Limits for Dobot M1 Pro
    J1_MIN, J1_MAX = -85.0, 85.0
    J2_MIN, J2_MAX = -135.0, 135.0
    Z_MIN, Z_MAX = 5.0, 245.0

    # --- COORDINATE CHECK (Reverted Mode) ---
    if not STRICT_JOINT_CHECKING:
        # This will fail for any point where X or Y > 85
        if not (-85.0 <= x <= 85.0 and -135.0 <= y <= 135.0):
            raise ValueError(f"Target ({x}, {y}) blocked by X/Y coordinate check (reverted mode)")

    # Inverse kinematics calculations
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1:
        raise ValueError("Target position out of reach")
    D = max(-1,min(1,D))
    theta2 = math.atan2(math.sqrt(1 - D**2), D)
    theta1 = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))

    # Convert radians to degrees
    j1 = math.degrees(theta1)
    j2 = math.degrees(theta2)

    if STRICT_JOINT_CHECKING:
        # check if orgional position is outside limits 
        if not (J1_MIN <= j1 <= J1_MAX and J2_MIN <= j2 <= J2_MAX):
            # try flipping the elbow
            theta2_alt = math.atan2(-math.sqrt(1- D**2),D)
            theta1_alt = math.atan2(y,x) - math.atan2(L2* math.sin(theta2_alt), L1 + L2 * math.cos(theta2_alt))
            j1_alt , j2_alt = math.degrees(theta1_alt) , math.degrees(theta2_alt)
            # checks if flipping the elbow works
            if ( J1_MIN <= j1_alt <= J1_MAX and J2_MIN <= j2_alt <= J2_MAX):
                j1 , j2 = j1_alt,j2_alt
            else:
                # if both fail raise a value error
                raise ValueError(f"No Valid joint configuations within limits for  ({x},{y})")
    # even though this is mainly for the new mode this also checks for old just in case 
    if not (J1_MIN <= j1 <= J1_MAX):
        raise ValueError(f"J1 angle ({j1:.1f}°) exceeds hardware limit")
    if not (J2_MIN <= j2 <= J2_MAX):
        raise ValueError(f"J2 angle ({j2:.1f}°) exceeds hardware limit")

    if not (Z_MIN <= z <= Z_MAX):
        raise ValueError(f"Z height ({z}) out of range")

    # FIX: Return a list containing the solution list to satisfy the 'sols[0]' unpacking
    return [[j1, j2, z, r]]

# Example Usage:
# If you want to use the Cathedral points (which are > 85), 
# you will need to set STRICT_JOINT_CHECKING = True at the top.

# ---- Robot Control Function ----
def move_to_point(x, y, z=200, r=0):
    if is_jogging:
        messagebox.showwarning("Robot Busy", "Cannot send move command while jogging!")
        return
    
    
    try:
        sols = Ikinematics(x, y, z, r)
        
        if not sols:
            print(f"Target ({x}, {y}) is unreachable.")
            return

        # Now this unpacking will work because sols is [[...]]
        j1, j2, z_target, r_target = sols[0]

        if ROBOT_CONNECTED and robot:
            robot.dashboard.enable()
            print(f"Moving to ({x},{y}) | Joints: J1={j1:.1f}°, J2={j2:.1f}°")
            robot.movement.joint_to_joint_move([j1, j2, z_target, r_target])
            # --- ADD THESE TWO LINES TO SYNC ---
            m_x, m_y, m_z = x, y, z 
            # -----------------------------------
        else:
            print(f"DEMO MODE: Target ({x}, {y}) -> Joints J1={j1:.1f}°, J2={j2:.1f}°")
            # --- ADD THESE TWO LINES TO SYNC ---
            m_x, m_y, m_z = x, y, z 
            # -----------------------------------
    except Exception as e:
        print(f"Robot command failed: {e}")


# --- MANUAL CONTROL STATE ---
m_x, m_y, m_z = 250.0, 0.0, 200.0 
m_claw = 0 

# --- NEW: Jogging Handlers ---
# --- NEW AREA B: CONTINUOUS JOG HANDLERS ---
# --- REFINED AREA B ---
def handle_jog_press(axis_cmd):
    global is_jogging
    # Check 1: Is robot actually connected?
    # Check 2: Are we already jogging? (Prevents Windows key-repeat spam)
    if not ROBOT_CONNECTED or is_jogging or not manual_active.get():
        return
        
    # Get current joints from the background thread's latest data
    current_j = robot_data["joints"]
    
    # Send the safe command
    error = robot.movement.safe_move_jog(axis_cmd, current_j)
    
    if not error:
        is_jogging = True

def handle_jog_release(event):
    global is_jogging
    if ROBOT_CONNECTED:
        robot.movement.safe_move_jog("stop", [])
        is_jogging = False



def handle_manual_z(dz):
    if not manual_active.get(): return
    global m_z
    m_z = max(5.0, min(245.0, m_z + dz))
    move_to_point(m_x, m_y, m_z)

def handle_manual_claw():
    if not manual_active.get(): return
    global m_claw
    m_claw = 1 if m_claw == 0 else 0
    if ROBOT_CONNECTED and robot:
        robot.dashboard.set_digital_output(17, m_claw)
        print("Robot told to toggle claw")
        print(m_claw)
    claw_overdrive_btn.config(text=f"Claw: {'ON' if m_claw else 'OFF'}", bg="green" if m_claw else "red")

limit = 450
x = np.linspace(-limit, limit, 1000)
y = np.linspace(-limit, limit, 1000)
X, Y = np.meshgrid(x, y)

# Precompute constants
tan_85 = np.tan(np.radians(85))
tan_100 = np.tan(np.radians(100))

# Region definitions
r_squared = X**2 + Y**2
region1 = (153**2 <= r_squared) & (r_squared <= 400**2) & (np.abs(X) <= tan_85 * Y)
region2 = ((200 - X)**2 + (abs(200)/tan_85 - Y)**2 <= 200**2) & (Y <= -tan_100 * (X - 153) + 200/tan_85)
region3 = ((200 + X)**2 + (abs(200)/tan_85 - Y)**2 <= 200**2) & (Y <= tan_100 * (X + 153) + 200/tan_85)
final_region = region1 | (region2 | region3)

# Function to check if a point is inside the region
def is_inside(px, py):
    cond1 = (153**2 <= px**2 + py**2 <= 400**2) and (abs(px) <= tan_85 * py)
    cond2 = ((200 - px)**2 + (abs(200)/tan_85 - py)**2 <= 200**2) and (py <= -tan_100 * (px - 153) + 200/tan_85)
    cond3 = ((200 + px)**2 + (abs(200)/tan_85 - py)**2 <= 200**2) and (py <= tan_100 * (px + 153) + 200/tan_85)
    return cond1 or cond2 or cond3

# Lists to store valid and invalid points (now with z-values and claw state)
valid_points = []  # Will store tuples of (px, py, z, claw_state)
valid_scatters = []

# Global references for Joint Entry boxes
j1_entry = None
j2_entry = None
zj_entry = None

root = tk.Tk()
root.title("Robotic Arm Control")

# INITIALIZE HERE - This prevents the "Too early" error
global manual_active
manual_active = tk.BooleanVar(value=False)

# Create main container
main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=1)

# Create left side container for plot and manual input
left_container = tk.Frame(main_container)
left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Manual input frame (above the plot)
manual_frame = tk.Frame(left_container)
manual_frame.pack(fill=tk.X, padx=5, pady=5)


# --- MANUAL OVERDRIVE UI ---
overdrive_frame = tk.LabelFrame(manual_frame, text="Keyboard & Manual Overdrive", padx=10, pady=10)
overdrive_frame.pack(fill=tk.X, padx=10, pady=5)

# Safety Toggle (Must be ON for keys/buttons to work)
tk.Checkbutton(overdrive_frame, text="Enable Keyboard Control", variable=manual_active, 
               font=("Arial", 10, "bold"), fg="darkblue").grid(row=0, column=0, columnspan=3, pady=5)

# Z Control Buttons
tk.Button(overdrive_frame, text="Z Up (W)", width=10, command=lambda: handle_manual_z(10)).grid(row=1, column=0, padx=5)
tk.Button(overdrive_frame, text="Z Down (S)", width=10, command=lambda: handle_manual_z(-10)).grid(row=1, column=1, padx=5)

# Claw Toggle Button
claw_overdrive_btn = tk.Button(overdrive_frame, text="Claw: OFF", width=12, bg="red", fg="white", 
                               command=handle_manual_claw)
claw_overdrive_btn.grid(row=1, column=2, padx=5)

# Main Title
tk.Label(manual_frame, text="Manual Control Interface", font=("Arial", 12, "bold")).pack(pady=5)

# --- ROW 1: MANUAL POINT INPUT (XYZ) ---
input_fields_frame = tk.Frame(manual_frame)
input_fields_frame.pack(pady=5)

# X input
x_frame = tk.Frame(input_fields_frame)
x_frame.pack(side=tk.LEFT, padx=10)
tk.Label(x_frame, text="X (mm):").pack()
x_manual_entry = tk.Entry(x_frame, width=8)
x_manual_entry.pack()

# Y input
y_frame = tk.Frame(input_fields_frame)
y_frame.pack(side=tk.LEFT, padx=10)
tk.Label(y_frame, text="Y (mm):").pack()
y_manual_entry = tk.Entry(y_frame, width=8)
y_manual_entry.pack()

# Z input
z_frame = tk.Frame(input_fields_frame)
z_frame.pack(side=tk.LEFT, padx=10)
tk.Label(z_frame, text="Z (mm):").pack()
z_manual_entry = tk.Entry(z_frame, width=8)
z_manual_entry.insert(0, "200")
z_manual_entry.pack()

# Claw control (XYZ Row)
claw_frame = tk.Frame(input_fields_frame)
claw_frame.pack(side=tk.LEFT, padx=10)
tk.Label(claw_frame, text="Claw:").pack()
claw_var = tk.IntVar(value=0)
claw_radio_frame = tk.Frame(claw_frame)
claw_radio_frame.pack()
tk.Radiobutton(claw_radio_frame, text="OFF", variable=claw_var, value=0).pack(side=tk.LEFT)
tk.Radiobutton(claw_radio_frame, text="ON", variable=claw_var, value=1).pack(side=tk.LEFT)

# Add Point Button (Inline with Row 1)
add_manual_button = tk.Button(input_fields_frame, text="Add Point", 
                              command=lambda: add_manual_point(), 
                              bg="lightgreen", padx=20)
add_manual_button.pack(side=tk.LEFT, padx=20)

# --- Visual Separator ---
tk.Frame(manual_frame, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=10)

# --- ROW 2: MANUAL JOINT CONTROL (J1, J2, Z) ---
joint_fields_frame = tk.Frame(manual_frame)
joint_fields_frame.pack(pady=5)

# J1 Entry
f1 = tk.Frame(joint_fields_frame); f1.pack(side=tk.LEFT, padx=10)
tk.Label(f1, text="J1 (deg):").pack()
j1_entry = tk.Entry(f1, width=8); j1_entry.pack()

# J2 Entry
f2 = tk.Frame(joint_fields_frame); f2.pack(side=tk.LEFT, padx=10)
tk.Label(f2, text="J2 (deg):").pack()
j2_entry = tk.Entry(f2, width=8); j2_entry.pack()

# Z (Joint) Entry
f3 = tk.Frame(joint_fields_frame); f3.pack(side=tk.LEFT, padx=10)
tk.Label(f3, text="Z (mm):").pack()
zj_entry = tk.Entry(f3, width=8); zj_entry.insert(0, "200"); zj_entry.pack()

# Claw control (Joint Row)
claw_frame_j = tk.Frame(joint_fields_frame)
claw_frame_j.pack(side=tk.LEFT, padx=10)
tk.Label(claw_frame_j, text="Claw:").pack()
claw_var_j = tk.IntVar(value=0)
claw_radio_frame_j = tk.Frame(claw_frame_j)
claw_radio_frame_j.pack()
tk.Radiobutton(claw_radio_frame_j, text="OFF", variable=claw_var_j, value=0).pack(side=tk.LEFT)
tk.Radiobutton(claw_radio_frame_j, text="ON", variable=claw_var_j, value=1).pack(side=tk.LEFT)

# Move Joints Button (Inline with Row 2)
move_j_btn = tk.Button(joint_fields_frame, text="Move Joints", 
                       command=lambda: manual_joint_move(), 
                       bg="lightblue", padx=20)
move_j_btn.pack(side=tk.LEFT, padx=20)

# Create a graph to plot the valid region
fig, ax = plt.subplots(figsize=(4,4))
ax.set_title("Arm Valid Region")
fig.tight_layout()

#set x and y axis limits, with 100s interval ticks and 50s minor ticks
ax.set_xlim(-450, 450)
ax.set_ylim(-250, 450)
ax.set_xticks(np.arange(-400, 401, 100))
ax.set_yticks(np.arange(-300, 401, 100))
ax.grid(which='major', linestyle='-', linewidth=0.8)
ax.set_xticks(np.arange(-450, 451, 50), minor=True)
ax.set_yticks(np.arange(-300, 451, 50), minor=True)
ax.grid(which='minor', linestyle='--', linewidth=0.5)
ax.set_aspect('equal', 'box') 

# Setup the valid region plot in light grey
ax.contourf(X, Y, final_region, levels=[0.5, 1], colors=['lightgrey'], alpha=0.5)

# Embed matplotlib figure into Tkinter
canvas = FigureCanvasTkAgg(fig, master=left_container)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# Frame for valid points list and buttons
frame = tk.Frame(main_container)
frame.pack(side=tk.RIGHT, fill=tk.Y)

# Connection status indicator
status_frame = tk.Frame(frame)
status_frame.pack(pady=5)
status_color = "green" if ROBOT_CONNECTED else "red"
status_text = "Robot Connected" if ROBOT_CONNECTED else "Demo Mode (No Robot)"
status_label = tk.Label(status_frame, text=status_text, fg=status_color, font=("Arial", 10, "bold"))
status_label.pack()

tk.Label(frame, text="Valid Points (FIFO)").pack(pady=(10,0))
points_listbox = tk.Listbox(frame, width=30, height=25)
points_listbox.pack(fill=tk.BOTH, expand=1)

# Function to add manual point
def add_manual_point():
    try:
        # Get values from input fields
        x_val = float(x_manual_entry.get())
        y_val = float(y_manual_entry.get())
        z_val = float(z_manual_entry.get())
        claw_state = claw_var.get()  # Get claw state from radio buttons
        
        # Validate z-value range
        if not (5.0 <= z_val <= 245.0):
            messagebox.showerror("Invalid Z-Value", "Z-value must be between 5 and 245 mm")
            return
        
        # Check if point is in valid region
        if is_inside(x_val, y_val):
            # Add valid point with its z-value and claw state
            valid_points.append((x_val, y_val, z_val, claw_state))
            scatter = ax.scatter(x_val, y_val, color='blue', s=50, marker='s')  # Blue square for manual points
            valid_scatters.append(scatter)
            
            claw_text = "ON" if claw_state == 1 else "OFF"
            points_listbox.insert(tk.END, f"{len(valid_points)}: ({x_val:.2f}, {y_val:.2f}, z={z_val:.1f}, claw={claw_text}) [Manual]")
            canvas.draw()
            
            # Clear input fields after successful addition
            x_manual_entry.delete(0, tk.END)
            y_manual_entry.delete(0, tk.END)
            z_manual_entry.delete(0, tk.END)
            z_manual_entry.insert(0, "200")  # Reset z to default
            # Keep claw setting as is (don't reset)
            
            print(f"Manual point added: ({x_val:.2f}, {y_val:.2f}, z={z_val:.1f}, claw={claw_text})")
        else:
            messagebox.showerror("Invalid Point", f"Point ({x_val:.2f}, {y_val:.2f}) is outside the valid region")
            
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for X, Y, and Z coordinates")

# Function to remove first valid point (FIFO)
def remove_first_point():
    if valid_points:
        # Remove point from data
        valid_points.pop(0)
        # Remove scatter plot
        scatter = valid_scatters.pop(0)
        scatter.remove()
        # Remove from listbox
        points_listbox.delete(0)
        canvas.draw()

def add_dobot_instructions():
    if not valid_points:
        messagebox.showwarning("No Points", "No valid points to send to robot!")
        return
    
    def process_next_point():
        if valid_points:
            # Get the first point with its individual z-value and claw state
            px, py, point_z, claw_state = valid_points[0]
            x = round(py, 2)
            y = -1 * round(px, 2)
            if(claw_state == 1):
                claw_text = "ON"
            else:
                claw_text = "OFF"
            print(f"Sending point to robot: x={px:.2f}, y={py:.2f}, z={point_z:.2f}, claw={claw_text}")
            
            # Send point to robot with the point's specific z-value
            try:
                move_to_point(x, y, point_z, 0)
                
                # Control claw using digital output
                if ROBOT_CONNECTED and robot:
                    if(claw_text == "ON"):
                        robot.dashboard.set_digital_output(17, 1)
                        sleep(1)  # Ensure claw has time to activate
                        robot.dashboard.set_digital_output(17, 0)
                    else:
                        robot.dashboard.set_digital_output(17, 1)
                        sleep(1)  # Ensure claw has time to deactivate
                        robot.dashboard.set_digital_output(17, 0)
                    print(f"Claw set to {claw_text}")
                else:
                    print(f"DEMO MODE: Would set claw to {claw_text}")
                    
            except Exception as e:
                print(f"Robot command failed: {e}")
                messagebox.showerror("Robot Error", f"Failed to send command to robot: {e}")
                return
            
            # Remove the point from GUI and internal lists
            remove_first_point()
            # Schedule the next point after 3 seconds
            root.after(3000, process_next_point)

    process_next_point()

def dobot_error_reset():
    if ROBOT_CONNECTED and robot:
        try:
            robot.dashboard.clear_errors()
            print("Robot errors cleared")
        except Exception as e:
            print(f"Failed to clear robot errors: {e}")
            messagebox.showerror("Robot Error", f"Failed to clear errors: {e}")
    else:
        print("DEMO MODE: Would clear robot errors")

def add_test_points_from_list():
    """Open a dialog to input a list of test points for testing purposes."""
    dialog = tk.Toplevel(root)
    dialog.title("Add Test Points")
    dialog.geometry("500x400")
    dialog.transient(root)
    dialog.grab_set()

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
    y = (dialog.winfo_screenheight() // 2) - (400 // 2)
    dialog.geometry(f"500x400+{x}+{y}")

    # Title
    tk.Label(dialog, text="Enter Test Points List", font=("Arial", 14, "bold")).pack(pady=10)

    # Instructions
    instructions = tk.Label(dialog, text="Enter a Python list of points.\nFormat: [(x, y, z, claw), (x, y, z, claw), ...]\nExample: [(100, 50, 200, 0), (150, 75, 180, 1)]", justify=tk.LEFT)
    instructions.pack(pady=5, padx=10)

    # Text area for input
    text_frame = tk.Frame(dialog)
    text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    text_input = tk.Text(text_frame, height=10, width=50)
    text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(text_frame, command=text_input.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_input.config(yscrollcommand=scrollbar.set)

    # Example button
    def insert_example():
        example = """[(100, 50, 200, 0), (150, 75, 180, 1), (200, 100, 220, 0)]"""
        text_input.delete(1.0, tk.END)
        text_input.insert(1.0, example)

    tk.Button(dialog, text="Insert Example", command=insert_example).pack(pady=5)

    # Results label
    result_label = tk.Label(dialog, text="", fg="blue")
    result_label.pack(pady=5)

    def parse_and_add_points():
        input_text = text_input.get(1.0, tk.END).strip()

        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter a list of points")
            return

        try:
            # Try to evaluate the input as a Python literal
            import ast
            points_list = ast.literal_eval(input_text)

            if not isinstance(points_list, list):
                messagebox.showerror("Invalid Format", "Input must be a list")
                return

            if len(points_list) == 0:
                messagebox.showwarning("Empty List", "The list is empty")
                return

            # Validate each point
            valid_count = 0
            invalid_count = 0

            for i, point in enumerate(points_list):
                try:
                    if len(point) != 4:
                        print(f"Point {i+1}: Invalid number of values (expected 4, got {len(point)})")
                        invalid_count += 1
                        continue

                    px, py, pz, claw = point

                    # Validate types and ranges
                    if not all(isinstance(coord, (int, float)) for coord in [px, py, pz]):
                        print(f"Point {i+1}: X, Y, Z must be numbers")
                        invalid_count += 1
                        continue

                    if not isinstance(claw, int) or claw not in [0, 1]:
                        print(f"Point {i+1}: Claw must be 0 (OFF) or 1 (ON)")
                        invalid_count += 1
                        continue

                    if not (5.0 <= pz <= 245.0):
                        print(f"Point {i+1}: Z-value must be between 5 and 245 mm")
                        invalid_count += 1
                        continue

                    if not is_inside(px, py):
                        print(f"Point {i+1}: ({px:.2f}, {py:.2f}) is outside valid region")
                        invalid_count += 1
                        continue

                    # Add valid point
                    valid_points.append((px, py, pz, claw))
                    scatter = ax.scatter(px, py, color='purple', s=50, marker='D')  # Purple diamond for test points
                    valid_scatters.append(scatter)

                    claw_text = "ON" if claw == 1 else "OFF"
                    points_listbox.insert(tk.END, f"{len(valid_points)}: ({px:.2f}, {py:.2f}, z={pz:.1f}, claw={claw_text}) [Test]")
                    valid_count += 1

                except Exception as e:
                    print(f"Point {i+1}: Error - {e}")
                    invalid_count += 1
                    continue

            # Update plot
            canvas.draw()

            # Show results
            result_text = f"Added {valid_count} valid points"
            if invalid_count > 0:
                result_text += f", {invalid_count} invalid points skipped"
            result_label.config(text=result_text)

            if valid_count > 0:
                print(f"Successfully added {valid_count} test points")
                if invalid_count > 0:
                    print(f"Skipped {invalid_count} invalid points")
            else:
                messagebox.showwarning("No Valid Points", "No valid points were added. Check the console for details.")

        except (ValueError, SyntaxError) as e:
            messagebox.showerror("Parse Error", f"Invalid Python syntax: {e}\n\nMake sure to use proper list format with brackets and parentheses.")

    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=10, fill=tk.X)

    tk.Button(button_frame, text="Add Points", command=parse_and_add_points, bg="lightgreen").pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Cancel", command=dialog.destroy, bg="lightcoral").pack(side=tk.LEFT, padx=10)

    # Wait for dialog to close
    dialog.wait_window()

# Button to remove first point
remove_button = tk.Button(frame, text="Remove First Point (FIFO)", command=remove_first_point)
remove_button.pack(pady=5)

# Button to send coordinates to dobot
send_button = tk.Button(frame, text="Send Instructions", command=add_dobot_instructions, bg="lightgreen")
send_button.pack(pady=5)

# Button to reset error
error_button = tk.Button(frame, text="Clear Errors", command=dobot_error_reset, bg="orange")
error_button.pack(pady=5)

# Button to add test points from list
test_points_button = tk.Button(frame, text="Add Test Points", command=add_test_points_from_list, )
test_points_button.pack(pady=5)

# Custom dialog for Z-value and claw state

def get_point_settings(px, py):
    # Create the popup window
    dialog = tk.Toplevel(root)
    dialog.title("Point Settings")
    dialog.geometry("300x250")
    dialog.transient(root)
    dialog.grab_set()  # Forces user to interact with this window before the main one
    
    # Center the dialog on screen
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
    y = (dialog.winfo_screenheight() // 2) - (250 // 2)
    dialog.geometry(f"300x250+{x}+{y}")
    
    # Initialize the result dictionary
    result = {'z': None, 'claw': 0}
    
    # CRITICAL: This variable tells the code when the "Add Point" button is clicked
    submitted = tk.BooleanVar(value=False)
    
    # UI Elements
    tk.Label(dialog, text=f"Settings for point:", font=("Arial", 10, "bold")).pack(pady=5)
    tk.Label(dialog, text=f"X: {px:.2f}, Y: {py:.2f}", font=("Arial", 9)).pack()
    
    # Z-value input
    tk.Label(dialog, text="\nZ-value (5-245 mm):").pack()
    z_entry = tk.Entry(dialog, width=15)
    z_entry.insert(0, "200") # Default height
    z_entry.pack()
    
    # Claw control
    tk.Label(dialog, text="\nClaw State:").pack()
    claw_var_inner = tk.IntVar(value=0)
    radio_frame = tk.Frame(dialog)
    radio_frame.pack()
    tk.Radiobutton(radio_frame, text="OFF", variable=claw_var_inner, value=0).pack(side=tk.LEFT)
    tk.Radiobutton(radio_frame, text="ON", variable=claw_var_inner, value=1).pack(side=tk.LEFT)
    
    # Internal function for the button click
    def add_point_clicked():
        try:
            z_val = float(z_entry.get())
            if 5.0 <= z_val <= 245.0:
                # SAVE the values into our result dictionary
                result['z'] = z_val
                result['claw'] = claw_var_inner.get()
                
                # Signal that we are done and close window
                submitted.set(True)
                dialog.destroy()
            else:
                messagebox.showerror("Invalid Z", "Z must be between 5 and 245")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a numeric Z-value")
    
    def cancel_clicked():
        # result['z'] remains None, so the point won't be saved
        submitted.set(True) # Set to true just to break the wait loop
        dialog.destroy()

    # Buttons
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Add Point", command=add_point_clicked, bg="lightgreen", width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Cancel", command=cancel_clicked, bg="lightcoral", width=10).pack(side=tk.LEFT, padx=5)
    
    # CRITICAL: This pauses the main script until 'submitted' is set to True
    # Without this, the function returns result={'z':None} immediately.
    root.wait_variable(submitted)
    
    return result

# Event handler for clicking on the plot
def onclick(event):

    # --- ADD THIS CHECK AT THE VERY TOP ---
    global is_jogging
    if is_jogging:
        print("Click ignored: Robot is currently jogging.")
        return 
    # --------------------------------------

    if event.xdata is None or event.ydata is None:
        return
    px, py = event.xdata, event.ydata
    if is_inside(px, py):
        # Get point settings (z-value and claw state)
        settings = get_point_settings(px, py)
        
        if settings['z'] is not None:  # User didn't cancel
            # Add valid point with its z-value and claw state
            valid_points.append((px, py, settings['z'], settings['claw']))
            scatter = ax.scatter(px, py, color='green', s=50)
            valid_scatters.append(scatter)
            
            claw_text = "ON" if settings['claw'] == 1 else "OFF"
            points_listbox.insert(tk.END, f"{len(valid_points)}: ({px:.2f}, {py:.2f}, z={settings['z']:.1f}, claw={claw_text})")
        # If user cancelled, don't add the point
    else:
        # Add invalid point and remove after 1 second
        scatter = ax.scatter(px, py, color='red', s=50)
        canvas.draw()
        root.after(1000, lambda: remove_invalid_point(scatter))

    canvas.draw()

def remove_invalid_point(scatter):
    scatter.remove()
    canvas.draw()

def manual_joint_move():
    global robot, ROBOT_CONNECTED
    
    # physical limits for M1 Pro
    J1_MIN, J1_MAX = -85.0, 85.0
    J2_MIN, J2_MAX = -135.0, 135.0
    Z_MIN, Z_MAX = 5.0, 245.0
    J4_FIXED = -35.0 

    try:
        j1 = float(j1_entry.get())
        j2 = float(j2_entry.get())
        z = float(zj_entry.get())

        if not (J1_MIN <= j1 <= J1_MAX and J2_MIN <= j2 <= J2_MAX and Z_MIN <= z <= Z_MAX):
            messagebox.showerror("Out of Range", "Joint values outside limits!")
            return

        if ROBOT_CONNECTED and robot:
            robot.movement.joint_to_joint_move([j1, j2, z, J4_FIXED])
        else:
            messagebox.showinfo("Demo Mode", f"Moving to J1:{j1} J2:{j2} Z:{z}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for joints.")


# Create a global variable for the 'live' marker so we can move it
live_robot_dot = ax.plot([], [], 'ro', markersize=10, label='Live Robot')[0]

def update_gui_from_feedback():
    """Refreshes the plot and labels with the robot's actual position."""
    if ROBOT_CONNECTED:
        # 1. Get Cartesian X, Y from the real-time data
        curr_x = robot_data["cartesian"][0]
        curr_y = robot_data["cartesian"][1]
        
        # 2. Update the red dot on the plot
        live_robot_dot.set_data([curr_x], [curr_y])
        
        # 3. Redraw only the idle parts of the canvas (prevents lag)
        fig.canvas.draw_idle() 

    # Schedule this function to run again in 100ms (10 times per second)
    root.after(100, update_gui_from_feedback)



# Connect the click event
fig.canvas.mpl_connect('button_press_event', onclick)

# Add instructions label
instructions = tk.Label(frame, text="Instructions:\n1. Click points on plot OR\n2. Use manual input (X,Y,Z) OR\n3. Use 'Add Test Points' for batch testing\n4. Send to robot",
                       justify=tk.LEFT, font=("Arial", 9), bg="lightyellow")
instructions.pack(pady=10)

# --- KEYBOARD BINDINGS ---
# --- NEW AREA C: JOGGING BINDINGS ---

# J1 Control (Shoulder)
root.bind("<KeyPress-Up>",    lambda e: handle_jog_press("J1+"))
root.bind("<KeyRelease-Up>",  handle_jog_release)

root.bind("<KeyPress-Down>",  lambda e: handle_jog_press("J1-"))
root.bind("<KeyRelease-Down>", handle_jog_release)

# J2 Control (Elbow)
root.bind("<KeyPress-Left>",  lambda e: handle_jog_press("J2+"))
root.bind("<KeyRelease-Left>", handle_jog_release)

root.bind("<KeyPress-Right>", lambda e: handle_jog_press("J2-"))
root.bind("<KeyRelease-Right>", handle_jog_release)

# J3 Control (Z-Axis Height)
root.bind("<KeyPress-w>",     lambda e: handle_jog_press("J3+"))
root.bind("<KeyRelease-w>",   handle_jog_release)

root.bind("<KeyPress-s>",     lambda e: handle_jog_press("J3-"))
root.bind("<KeyRelease-s>",   handle_jog_release)



update_gui_from_feedback()
root.mainloop()



