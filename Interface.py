import customtkinter # Library for Desktop app

class InfoFrame(customtkinter.CTkFrame):
    def __init__(self, master, point_frame):
        super().__init__(master)
        self.point_frame = point_frame

        self.grid_columnconfigure((0,1,2,3,4,5), weight=1)

        # Title
        self.title = customtkinter.CTkLabel(self, text="Adress", fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=6)

        self.title = customtkinter.CTkLabel(self, text="Robot IP:")
        self.title.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.x_input = customtkinter.CTkEntry(self, placeholder_text="192.168.1.6")
        self.x_input.grid(row=1, column=1, padx=10, pady=5, sticky="ew", columnspan=2)

        # Title
        self.title = customtkinter.CTkLabel(self, text="Inputs", fg_color="gray30", corner_radius=6)
        self.title.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="ew", columnspan=6)
        # Upload Button
        self.upload_button = customtkinter.CTkButton(self, text="Confirm", command=self.upload)
        self.upload_button.grid(row=1, column=5, padx=10, pady=10, sticky="ew")

        # X, Y, Z Inputs
        self.x_input = customtkinter.CTkEntry(self, placeholder_text="X")
        self.x_input.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.y_input = customtkinter.CTkEntry(self, placeholder_text="Y")
        self.y_input.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        self.z_input = customtkinter.CTkEntry(self, placeholder_text="Z")
        self.z_input.grid(row=3, column=2, padx=10, pady=5, sticky="ew")

        # Radio Buttons for Open/Close
        self.state_var = customtkinter.StringVar(value="open")
        self.open_radio = customtkinter.CTkRadioButton(self, text="Open", variable=self.state_var, value="open")
        self.open_radio.grid(row=3, column=3, padx=10, pady=5, sticky="w")

        self.close_radio = customtkinter.CTkRadioButton(self, text="Close", variable=self.state_var, value="close")
        self.close_radio.grid(row=3, column=4, padx=10, pady=5, sticky="w")

        # Upload Button
        self.upload_button = customtkinter.CTkButton(self, text="Upload", command=self.upload)
        self.upload_button.grid(row=3, column=5, padx=10, pady=10, sticky="ew")

    def upload(self):
        x = self.x_input.get()
        y = self.y_input.get()
        z = self.z_input.get()
        state = self.state_var.get()

        if x and y and z:
            command_text = f"({x}, {y}, {z}) -- ({state})"
            self.point_frame.add_command(command_text)


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
