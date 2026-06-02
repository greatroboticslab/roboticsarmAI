import customtkinter # Library for Desktop app
import main             # Fixes 'main is undefined' warning across the file


class InfoFrame(customtkinter.CTkFrame):
    def __init__(self, master, point_frame):
        super().__init__(master)
        self.point_frame = point_frame
        self.grid_columnconfigure((0,1,2,3,4,5), weight=1)

        # --- Row 0 & 1: Address ---
        self.title_addr = customtkinter.CTkLabel(self, text="Address", fg_color="gray30", corner_radius=6)
        self.title_addr.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=6)

        self.ip_label = customtkinter.CTkLabel(self, text="Robot IP:")
        self.ip_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.ip_input = customtkinter.CTkEntry(self, placeholder_text="192.168.1.6") # Renamed from x_input
        self.ip_input.grid(row=1, column=1, padx=10, pady=5, sticky="ew", columnspan=2)
        
        self.confirm_button = customtkinter.CTkButton(self, text="Confirm", command=self.upload)
        self.confirm_button.grid(row=1, column=5, padx=10, pady=10, sticky="ew")

        # --- Row 2 & 3: Standard Inputs ---
        self.title_inp = customtkinter.CTkLabel(self, text="Inputs", fg_color="gray30", corner_radius=6)
        self.title_inp.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=6)

        self.x_input = customtkinter.CTkEntry(self, placeholder_text="X")
        self.x_input.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.y_input = customtkinter.CTkEntry(self, placeholder_text="Y")
        self.y_input.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.z_input = customtkinter.CTkEntry(self, placeholder_text="Z")
        self.z_input.grid(row=3, column=2, padx=10, pady=5, sticky="ew")

        self.state_var = customtkinter.StringVar(value="open")
        self.open_radio = customtkinter.CTkRadioButton(self, text="Open", variable=self.state_var, value="open")
        self.open_radio.grid(row=3, column=3, padx=10, pady=5, sticky="w")
        self.close_radio = customtkinter.CTkRadioButton(self, text="Close", variable=self.state_var, value="close")
        self.close_radio.grid(row=3, column=4, padx=10, pady=5, sticky="w")

        self.upload_button = customtkinter.CTkButton(self, text="Upload", command=self.upload)
        self.upload_button.grid(row=3, column=5, padx=10, pady=10, sticky="ew")

        # --- Row 4, 5, 6: Manual Controls (NEW) ---
        self.title_man = customtkinter.CTkLabel(self, text="Manual Controls", fg_color="gray30", corner_radius=6)
        self.title_man.grid(row=4, column=0, padx=10, pady=(20, 0), sticky="ew", columnspan=6)

        self.up_btn = customtkinter.CTkButton(self, text="Z Up (W)", command=lambda: self.master.manual_z(10))
        self.up_btn.grid(row=5, column=0, padx=5, pady=10, sticky="ew", columnspan=2)
        self.down_btn = customtkinter.CTkButton(self, text="Z Down (S)", command=lambda: self.master.manual_z(-10))
        self.down_btn.grid(row=5, column=2, padx=5, pady=10, sticky="ew", columnspan=2)

        self.claw_active = False
        self.claw_btn = customtkinter.CTkButton(self, text="Claw: OFF", fg_color="darkred", command=self.toggle_claw_ui)
        self.claw_btn.grid(row=5, column=4, padx=5, pady=10, sticky="ew", columnspan=2)

        self.manual_enabled = customtkinter.BooleanVar(value=False)
        self.safety_switch = customtkinter.CTkSwitch(self, text="Enable Keyboard Control", variable=self.manual_enabled)
        self.safety_switch.grid(row=6, column=0, padx=10, pady=10, sticky="ew", columnspan=6)

    def toggle_claw_ui(self, *args):
        # Toggles the claw state and updates button text/color
        if self.claw_state:
            self.claw_state = False
            self.toggle_claw_btn.configure(text="Claw: Open", fg_color="green", hover_color="darkgreen")
            
            # Physical Hardware Actuation
            if main.ROBOT_CONNECTED and main.robot is not None:
                try:
                    main.robot.dashboard.set_tool_output(1, 1)
                    print("Hardware Command Sent: Forearm Tool Output 1 -> ON (Open)")
                except Exception:
                    pass
            else:
                print("Demo Mode: Claw Open state simulated.")
        else:
            self.claw_state = True
            self.toggle_claw_btn.configure(text="Claw: Closed", fg_color="red", hover_color="darkred")
            
            # Physical Hardware Actuation
            if main.ROBOT_CONNECTED and main.robot is not None:
                try:
                    main.robot.dashboard.set_tool_output(1, 0)
                    print("Hardware Command Sent: Forearm Tool Output 1 -> OFF (Closed)")
                except Exception:
                    pass
            else:
                print("Demo Mode: Claw Closed state simulated.")

                
    def upload(self):
        target_ip = self.ip_input.get() or "192.168.1.6" 
        success = main.initialize_robot(target_ip)
        if success:
            self.point_frame.add_command(f"System: {target_ip} Connected")
        else:
            self.point_frame.add_command("System: Demo Mode")

class GraphFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)

        # Title
        self.title = customtkinter.CTkLabel(self, text="Graph", fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")


class PointFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)

        # Title
        self.title = customtkinter.CTkLabel(self, text="Point Location", fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")

        # Listbox to store multiple commands
        self.grid_rowconfigure(1, weight=5)
        self.command_listbox = customtkinter.CTkTextbox(self, wrap="none")
        self.command_listbox.insert("end", "No commands yet\n")
        self.command_listbox.configure(state="disabled")
        self.command_listbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")

        # Send button (kept from original)
        self.sendButton = customtkinter.CTkButton(self, text="Send", fg_color="gray30", corner_radius=6)
        self.sendButton.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsew")

        # Clear All button
        self.clearButton = customtkinter.CTkButton(self, text="Clear All", fg_color="red", corner_radius=6, command=self.clear_commands)
        self.clearButton.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="nsew")

    def add_command(self, text):
        self.command_listbox.configure(state="normal")
        current_text = self.command_listbox.get("1.0", "end").strip()
        if current_text == "No commands yet":
            self.command_listbox.delete("1.0", "end")
        self.command_listbox.insert("end", text + "\n")
        self.command_listbox.configure(state="disabled")

    def clear_commands(self):
        self.command_listbox.configure(state="normal")
        self.command_listbox.delete("1.0", "end")
        self.command_listbox.insert("end", "No commands yet\n")
        self.command_listbox.configure(state="disabled")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # ====== App Appearance ======
        self.title("Robotic Arm manual-input control")
        self.geometry("900x900")

        # Grid config
        self.grid_rowconfigure((0), weight=1)
        self.grid_rowconfigure((1), weight=5)
        self.grid_columnconfigure((0), weight=5)
        self.grid_columnconfigure((1), weight=1)

        # ------ Right Column (PointFrame first, so we can pass it to InfoFrame) ------
        self.location_frame = PointFrame(self)
        self.location_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew", rowspan=2)

        # ------ Left Column ------
        self.info_frame = InfoFrame(self, self.location_frame)
        self.info_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.graph_frame = GraphFrame(self)
        self.graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")


app = App()
app.mainloop()
