# Robotic Arm Controller

A Python application that controls a robotic arm by inputting **Cartesian coordinates (X, Y, Z)**.

**Future version**: will allow control through **individual joint angles**, enabling movement by controller and later by AI.

## Features
- Move the robotic arm by specifying X, Y, Z coordinates
- Visualize coordinates on an XY plot
- Maintain a FIFO list of coordinates to be executed by the arm

## Setup and Installation
Follow these steps to set up the project and install all required dependencies.

1. **Clone the repository**
```powershell
git clone https://github.com/yourusername/robotic-arm-controller.git
cd robotic-arm-controller
```
2. **Create a virtual environment**
```powershell
python -m venv venv
```
3. **Activate the virtual environment**
```powershell
.\venv\Scripts\Activate.ps1
```
4. **Install Dependencies**
```powershell
pip install -r requirements.txt
```
5. **Run the Application**
```powershell
python main.py
```

## Using the GUI (while `main.py` is running)
When you start the application with `python main.py`, the GUI window will open and let you plot and send coordinate sequences to the robotic arm. Below are general usage instructions and recommended workflows.

- **Plotting points**: Use the plot area to add or visualise points in the XY plane. Depending on the GUI controls you can typically add points by clicking the plot or by entering exact coordinates in the input fields.
- **Adding coordinates to the list**: Enter coordinates using the provided input fields (X, Y, Z) and press the button to append them to the coordinate list. The list follows a FIFO protocol: the oldest coordinate will be executed first when you send the list to the arm.
- **Editing / Removing points**: Use the list controls to remove or reorder entries before sending. If the GUI supports it, select a point and delete or edit the values.
- **Sending to the arm**: There is a control/button to send the queued coordinate list to the robotic arm. Press it to begin execution. Monitor the GUI for status updates or errors.
- **Start / Stop / Pause**: Use the provided start/stop/pause controls to control execution. If your GUI does not have pause, stop will typically abort execution and clear or retain the queue depending on settings.

### Coordinate format
- Coordinates are Cartesian triples: `X, Y, Z` (units depend on your arm configuration, commonly millimeters).
- Example entry: `X: 150, Y: 0, Z: 100`
- Example list (pseudo-format):
```
[(150, 0, 100), (160, 20, 90), (140, -20, 95)]
```

### Typical workflow
1. Start the application: `python main.py` and wait for the GUI window.
2. Use the plot or input fields to add the desired waypoints to the list.
3. Verify the order of waypoints in the FIFO list. Edit or remove any points as needed.
4. Click `Send` (or equivalent) to begin execution on the robotic arm.
5. Monitor the status in the GUI and use `Stop` immediately if anything behaves unexpectedly.

### Safety & troubleshooting
- **Safety first**: Keep a safe distance from the arm during motion. Ensure no objects or people are inside the workspace when commanding moves.
- **Limits**: If a coordinate is outside the robot's reachable workspace, the arm may error or behave unpredictably. Validate coordinates before sending.
- **Connection errors**: If the GUI cannot communicate with the arm, check cables, power, and the device connection settings (serial port, IP, etc.). Restart the app after fixing connection issues.
- **Logs / Errors**: Check the terminal running `main.py` for log output and error messages — they usually provide clues for what went wrong.

## Example
1. Add three points: `(150, 0, 100)`, `(160, 20, 90)`, `(140, -20, 95)`.
2. Confirm order in the FIFO list.
3. Click `Send` to move the arm through the points in order.

If you'd like, I can update this section with exact button names and screenshots once you tell me what the GUI controls (or if you want, I can open `Interface.py` and extract the real control names for a precise README entry).

## Images
![GUI Screenshot](exanole_forARM.png)

![Control Box](control_box.jpg)

![Robot Back Panel](robot_back.jpg)

## Mac compatibility
This project works on macOS with these small adjustments. Use `python3` on macOS and create/activate the virtual environment as shown below.

1. Create venv:
```bash
python3 -m venv venv
```
2. Activate venv (macOS / Bash / Zsh):
```bash
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
python3 main.py
```

Notes for macOS users:
- If you use a conda environment, you can skip the `venv` steps and use `conda activate` instead.
- If any GUI libraries require a specific backend on macOS (e.g., for matplotlib), the terminal logs will indicate missing backends; install the required packages (for example, `pip install pyobjc-framework-Quartz` or follow the library-specific instructions).

## Network connection: Ethernet 2
To interact with the robot the PC must be connected to the robot controller over Ethernet. On Windows the GUI expects the network link on the machine's Ethernet adapter named or routed through `Ethernet 2` (this is how our environment is set up). On macOS and other platforms the same idea applies: connect the PC's Ethernet port directly to the robot controller (or to the switch on the same subnet) and ensure your machine's interface is on the same subnet as the robot.

Steps to configure the connection (generic):
1. Physically connect an Ethernet cable between your computer and the robot controller or switch.
2. Identify the local interface that corresponds to that port (Windows: `Ethernet 2`; macOS: likely `en0`, `en1`, or a Thunderbolt Ethernet adapter). Use System Preferences/Settings → Network or `ifconfig`/`ipconfig` to identify the interface.
3. Assign a static IP on the same subnet as the robot controller. Example (macOS):
```bash
# replace en0 with your interface name and choose an IP in the robot's subnet
sudo ifconfig en0 inet 192.168.1.100 netmask 255.255.255.0
```
On Windows use the GUI network settings to set a manual IPv4 address that matches the robot's subnet and use `Ethernet 2` as the active interface.
4. Connect from the GUI using the robot's IP address (enter it in the GUI's connection field or ensure the GUI is configured to use the same interface). If you don't know the robot's IP, check the robot controller documentation, or scan the local subnet with a network tool (for example `nmap`) to locate the device.

Troubleshooting tips:
- If the GUI shows "Demo Mode (No Robot)", check that the Ethernet interface is up and that the selected interface is the one connected to the robot.
- Ensure any OS firewall is disabled or configured to allow the GUI application to communicate on the local network.
- If you cannot reach the robot, try pinging the robot IP from a terminal to confirm connectivity.

If you'd like, I can extract exact connection fields and button names from `Interface.py` and update the README to show the exact GUI steps (I won't modify `Interface.py` itself unless you ask).
