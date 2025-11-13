# Robotic Arm Controller

A Python application that controls a robotic arm by inputting **Cartesian coordinates (X, Y, Z)**.  
Future version: will allow control through **individual joint angles**, enabling movement by controller and later by AI.

---

## Features
- Move the robotic arm by specifying X, Y, Z coordinates  
- Easy visualization of coordinates given:
  - Added a graph showing coordinate location along the xy-plane
  - Added a list of coordinates that follow a FIFO protocol

## Setup and Installation
Follow these steps to set up the project and install all required dependencies.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/robotic-arm-controller.git
   cd robotic-arm-controller
   ```

2. **Create a virtual enviroment**
Create a new isolated Python environment (recommended to keep dependencies clean).
  ```bash
  python3 -m venv venv
  ```

3. **Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```
4. **Run the Application**
  ```bash
  python main.py
  ```
