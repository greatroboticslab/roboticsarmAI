import logging as log
from typing import Optional
from .util import *
from .types import *


# TODO: The outer dashboard classes will implement all shared functionality, but the CR version will have additional methods.
# TODO: A better naming convention

# We leave it upon the caller of the API to see if they care about the errors.
# There should be no error handling inside of this API.
# This is just a faithful translation and communication layer.


class Dobot:
    def __init__(self, ip: str, urdf_file: Optional[URDF] = None, is_cr: bool = False, logging: bool = False, log_name: str = "output.log", log_level = log.DEBUG):
        if logging:
            log.basicConfig(filename=log_name, level=log_level)
        self.movement: Movement = Movement(ip, urdf_file)
        self.feedback: Feedback = Feedback(ip)
        self.dashboard: Dashboard = Dashboard(ip)
    

class Movement(DobotSocketConnection):
    def __init__(self, ip: str, urdf_file: URDF):
        super().__init__(ip, MOVEMENT_PORT)
        self.simulator = Simulator(urdf_file) if urdf_file else None
    
    # MovJ
    def move_joint(
        self, joints: list[float]) -> Optional[DobotError]:
        inner = ', '.join(str(joint) for joint in joints)
        command = f"MovJ({inner})"
        opt_error, ret_val = self.send_command(command)
        return opt_error

    # MovJIO
    def move_joint_io(
        self, x: float, y: float, z: float, rx: float, ry: float, rz: float, io_ports: list[IOPort]
    ) -> Optional[DobotError]:
        command = f"MovJIO({x}, {y}, {z}, {rx}, {ry}, {rz}"
        if io_ports:
            for io_port in io_ports:
                command += f",{{{io_port.mode}, {io_port.distance}, {io_port.index}, {io_port.status}}}" 
        command += ")"
        opt_error, ret_val = self.send_command(command)
        return opt_error

    # JointMovJ
    def joint_to_joint_move(self, joints: list[float]) -> Optional[DobotError]:
        inner = ', '.join(str(joint) for joint in joints)
        command = f"JointMovJ({inner})"
        opt_error, ret_val = self.send_command(command)
        return opt_error

    # MovL
    def move_linear(
        self, points: list[float]) -> Optional[DobotError]:
        inner = ', '.join(str(point) for point in points)
        command = f"MovL({inner})"
        opt_error, ret_val = self.send_command(command)
        return opt_error

    # MovLIO
    def move_linear_io(
        self, x: float, y: float, z: float, rx: float, ry: float, rz: float, io_ports: list[IOPort]
    ) -> Optional[DobotError]:
        command = f"MovLIO({x}, {y}, {z}, {rx}, {ry}, {rz}"
        if io_ports:
            for io_port in io_ports:
                command += f",{{{io_port.mode}, {io_port.distance}, {io_port.index}, {io_port.status}}}" 
        command += ")"
        opt_error, ret_val = self.send_command(command)
        return opt_error
    
    # Arc
    def move_arc(
        self, x: float, y: float, z: float, rx: float, ry: float, rz:float, x2:float, y2: float, z2: float, rx2: float, ry2: float, rz2:float
    ) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command(f"Arc({x}, {y}, {z}, {rx}, {ry}, {rz}, {x2}, {y2}, {z2}, {rx2}, {ry2}, {rz2})")
        return opt_error
    
    # Sync
    def sync(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("Sync()")
        return opt_error
    
    # RelMovJUser
    def relative_move_joint(self, offx: float, offy: float, offz: float, offrx: float, offry: float, offrz: float, user_index: int) -> Optional[DobotError]:
        user_index = clamp(user_index, 0, 9)
        opt_error, ret_val = self.send_command(f"RelMovJUser({offx}, {offy}, {offz}, {offrx}, {offry}, {offrz}, {user_index})")
        return opt_error

    # RelMovLUser
    def relative_linear_joint(self, offx: float, offy: float, offz: float, offrx: float, offry: float, offrz: float, user_index: int) -> Optional[DobotError]:
        user_index = clamp(user_index, 0, 9)
        opt_error, ret_val = self.send_command(f"RelMovLUser({offx}, {offy}, {offz}, {offrx}, {offry}, {offrz}, {user_index})")
        return opt_error

    # RelJointMovJ
    def relative_joint_motion(self, off1: float, off2: float, off3: float, off4: float, off5: float, off6: float) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command(f"RelJointMovJ({off1}, {off2}, {off3}, {off4}, {off5}, {off6})")
        return opt_error

    # MoveJog
    def move_jog(self, joint: JointSelection) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command(f"MovJog({joint})")
        return opt_error


class Dashboard(DobotSocketConnection):
    def __init__(self, ip: str):
        super().__init__(ip, DASHBOARD_PORT)

    def turn_on(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("PowerOn()")
        return opt_error

    def enable(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("EnableRobot()")
        return opt_error

    def disable(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("DisableRobot()")
        return opt_error

    def reset(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("ResetRobot()")
        return opt_error

    def clear_errors(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("ClearError()")
        return opt_error
    
    def clear_alarms(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("ClearAlarm()")
        return opt_error

    def emergency_stop(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("EmergencyStop()")
        return opt_error

    # TODO: Cover all possible Error IDs
    def get_error_id(self):
        opt_error, ret_val = self.send_command("GetErrorID()")
        log.info(f"Error IDs : {ret_val}")
        return opt_error

    def get_digital_input(self, index: int) -> DobotError | int:
        index = clamp(index, 1, 32)
        opt_error, ret_val = self.send_command(f"DI({index})")
        try:
            return int(ret_val)
        except:
            return opt_error

    def set_digital_output(self, index: int, val: int) -> Optional[DobotError]:
        index = clamp(index, 1, 16)
        val = clamp(val, 0, 1)
        opt_error, ret_val = self.send_command(f"DO({index}, {val})")
        return opt_error
    
    def robot_mode(self) -> DobotError | RobotMode:
        opt_error, ret_val = self.send_command("RobotMode()")
        if opt_error:
            return opt_error
        else:
            # NOTE: If this fails, then we have an uncovered error.
            return RobotMode(int(ret_val))
    
    def set_linear_accel(self, rate: int) -> Optional[DobotError]:
        rate = clamp(rate, 1, 100)
        opt_error, ret_val = self.send_command(f"AccL({rate})")
        return opt_error

    def set_joint_accel(self, rate: int) -> Optional[DobotError]:
        rate = clamp(rate, 1, 100)
        opt_error, ret_val = self.send_command(f"AccJ({rate})")
        return opt_error
    
    def set_linear_velocity(self, rate: int) -> Optional[DobotError]:
        rate = clamp(rate, 1, 100)
        opt_error, ret_val = self.send_command(f"SpeedL({rate})")
        return opt_error

    def set_joint_velocity(self, rate: int) -> Optional[DobotError]:
        rate = clamp(rate, 1, 100)
        opt_error, ret_val = self.send_command(f"SpeedJ({rate})")
        return opt_error

    def set_speedfactor(self, ratio: int) -> Optional[DobotError]:
        ratio = clamp(ratio, 1, 100)
        opt_error, ret_val = self.send_command(f"SpeedFactor({ratio})")
        return opt_error
    
    def set_arc_params(self, index: int) -> Optional[DobotError]:
        index = clamp(index, 0, 9)
        opt_error, ret_val = self.send_command(f"Arch({index})")
        return opt_error

    def set_continuous_path(self, ratio: int) -> Optional[DobotError]:
        ratio = clamp(ratio, 1, 100)
        opt_error, ret_val = self.send_command(f"CP({ratio})")
        return opt_error

    def set_user(self, index: int) -> Optional[DobotError]:
        index = clamp(index, 0, 9)
        opt_error, ret_val = self.send_command(f"User({index})")
        return opt_error

    def set_tool(self, index: int) -> Optional[DobotError]:
        index = clamp(index, 0, 9)
        opt_error, ret_val = self.send_command(f"Tool({index})")
        return opt_error
    
    def set_payload(self, weight: float, inertia: float) -> Optional[DobotError]:
        # TODO: Figure out acceptable ranges of weights and inertias for each model.
        opt_error, ret_val = self.send_command(f"PayLoad({weight, inertia})")
        return opt_error
    
    def set_digital_output(self, index: int, status: int) -> Optional[DobotError]:
        status = clamp(status, 0, 1)
        opt_error, ret_val = self.send_command(f"DO({index}, {status})")
        return opt_error
    
    def run_script(self, name: str) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command(f"RunScript({name})")
        return opt_error
    
    def stop_script(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("StopScript()")
        return opt_error

    def pause_script(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("PauseScript()")
        return opt_error

    def continue_script(self) -> Optional[DobotError]:
        opt_error, ret_val = self.send_command("ContinueScript()")
        return opt_error


# TODO: Create a numpy type which houses all needed values
# NOTE: Different for CR version
   
class Feedback(DobotSocketConnection):
    def __init__(self, ip: str):
        super().__init__(ip, REALTIME_FEEDBACK_PORT)
    
