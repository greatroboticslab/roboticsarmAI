import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, simpledialog
from time import sleep
import math

# Connect to the robot arm, if connection fails, run in demo mode
try:
    from dobot_util import Dobot
    robot = Dobot("192.168.1.6", logging=True)
    ROBOT_CONNECTED = True
    print("Robot connected successfully!")
except Exception as e:
    print(f"Robot connection failed: {e}")
    print("Running in demo mode - robot commands will be simulated")
    robot = None
    ROBOT_CONNECTED = False

# Calculate inverse kinematics for a 2-link planar arm
def Ikinematics(x, y, z=200.0, r=0.0):
    L1 = 200.0  # Length of first arm segment
    L2 = 200.0  # Length of second arm segment

    # Set Limitiations for the arm
    J1_min, J1_max = -85.0, 85.0
    J2_min, J2_max = -135.0, 135.0
    z_min, z_max = 5.0, 245.0

    # Check if the target position is within the arm's reach and limits
    if not (J1_min <= x <= J1_max and J2_min <= y <= J2_max and z_min <= z <= z_max):
        raise ValueError("Target position out of reach or exceeds joint limits")
    
    # Inverse kinematics calculations
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1:
        raise ValueError("Target position out of reach")
    theta2 = math.atan2(math.sqrt(1 - D**2), D)
    theta1 = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))

    # Convert radians to degrees
    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)

    # A list of parameters for the robot
    parameters = [theta1_deg, theta2_deg, z, 0]
    return parameters

# ---- Robot Control Function ----
def move_to_point(x, y, z=200, r=0):
    sols = Ikinematics(x, y, z, r)
    if ROBOT_CONNECTED and robot:
        robot.dashboard.enable()
        if sols:
            j1, j2, z, r = sols[0]  # choose first valid solution
            print(f"Moving to ({x},{y},{z},{r}) with joints J1={j1:.1f}°, J2={j2:.1f}°")
            robot.movement.joint_to_joint_move([j1,j2,z,r])
        else:
            print(f"Unreachable target: ({x},{y},{z},{r})")
            robot.dashboard.disable()
    else:
        if sols:
            j1, j2, z, r = sols[0]  # choose first valid solution
            print(f"DEMO MODE: Would move to ({x},{y},{z},{r}) with joints J1={j1:.1f}°, J2={j2:.1f}°")
        else:
            print(f"DEMO MODE: Unreachable target: ({x},{y},{z},{r})")

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

# Tkinter app setup
root = tk.Tk()
root.title("Arm Manual Control Interface")

# Create main container
main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=1)

# Create left side container for plot and manual input
left_container = tk.Frame(main_container)
left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Manual input frame (above the plot)
manual_frame = tk.Frame(left_container)
manual_frame.pack(fill=tk.X, padx=5, pady=5)

tk.Label(manual_frame, text="Manual Point Input", font=("Arial", 12, "bold"), ).pack(pady=5)

# Input fields frame
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
z_manual_entry.insert(0, "200")  # Default value
z_manual_entry.pack()

# Claw control frame
claw_frame = tk.Frame(input_fields_frame)
claw_frame.pack(side=tk.LEFT, padx=10)
tk.Label(claw_frame, text="Claw:").pack()

# Radio buttons for claw control
claw_var = tk.IntVar(value=0)  # Default to OFF (0)
claw_radio_frame = tk.Frame(claw_frame)
claw_radio_frame.pack()

tk.Radiobutton(claw_radio_frame, text="OFF", variable=claw_var, value=0, ).pack(side=tk.LEFT)
tk.Radiobutton(claw_radio_frame, text="ON", variable=claw_var, value=1, ).pack(side=tk.LEFT)

# Add manual point button
add_manual_button = tk.Button(input_fields_frame, text="Add Point", command=lambda: add_manual_point(), 
                             bg="lightgreen", padx=20)
add_manual_button.pack(side=tk.LEFT, padx=20)

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
                        robot.dashboard.set_digital_output(2, 1)
                        sleep(1)  # Ensure claw has time to activate
                        robot.dashboard.set_digital_output(2, 0)
                    else:
                        robot.dashboard.set_digital_output(1, 1)
                        sleep(1)  # Ensure claw has time to deactivate
                        robot.dashboard.set_digital_output(1, 0)
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
    dialog = tk.Toplevel(root)
    dialog.title("Point Settings")
    dialog.geometry("300x200")
    dialog.transient(root)
    dialog.grab_set()
    
    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
    y = (dialog.winfo_screenheight() // 2) - (200 // 2)
    dialog.geometry(f"300x200+{x}+{y}")
    
    result = {'z': None, 'claw': None}
    
    # Title
    tk.Label(dialog, text=f"Settings for point ({px:.2f}, {py:.2f})", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Z-value input
    z_frame = tk.Frame(dialog)
    z_frame.pack(pady=10)
    tk.Label(z_frame, text="Z-value (5-245 mm):").pack()
    z_entry = tk.Entry(z_frame, width=10)
    z_entry.insert(0, "200")
    z_entry.pack()
    
    # Claw control
    claw_frame = tk.Frame(dialog)
    claw_frame.pack(pady=10)
    tk.Label(claw_frame, text="Claw State:").pack()
    
    claw_dialog_var = tk.IntVar(value=0)
    radio_frame = tk.Frame(claw_frame)
    radio_frame.pack()
    tk.Radiobutton(radio_frame, text="OFF", variable=claw_dialog_var, value=0).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(radio_frame, text="ON", variable=claw_dialog_var, value=1).pack(side=tk.LEFT, padx=10)
    
    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=20)
    
    def add_point_clicked():
        try:
            z_val = float(z_entry.get())
            if 5.0 <= z_val <= 245.0:
                #result['z'] = z_val
                #result['claw'] = claw_dialog_var.get()
                dialog.destroy()
            else:
                messagebox.showerror("Invalid Z-Value", "Z-value must be between 5 and 245 mm")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric Z-value")
    
    def cancel_clicked():
        dialog.destroy()
    
    tk.Button(button_frame, text="Add Point", command=add_point_clicked, bg="lightgreen").pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Cancel", command=cancel_clicked, bg="lightcoral").pack(side=tk.LEFT, padx=10)
    
    # Wait for dialog to close
    dialog.wait_window()
    return result

# Event handler for clicking on the plot
def onclick(event):
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

# Connect the click event
fig.canvas.mpl_connect('button_press_event', onclick)

# Add instructions label
instructions = tk.Label(frame, text="Instructions:\n1. Click points on plot OR\n2. Use manual input (X,Y,Z) OR\n3. Use 'Add Test Points' for batch testing\n4. Send to robot",
                       justify=tk.LEFT, font=("Arial", 9), bg="lightyellow")
instructions.pack(pady=10)

root.mainloop()
