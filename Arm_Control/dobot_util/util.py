import socket
import logging as log
from pathlib import Path
from ikpy.chain import Chain
from typing import Optional, Tuple
from .types import DobotError, URDF

class DobotSocketConnection:
    def __init__(self, ip: str, port: int):
        self.socket = socket.socket()
        self.socket.settimeout(10.0)
        self.socket.connect((ip, port))
        log.debug("Connection established")

    " Sends a desired command over the socket connection and returns the potential error and return value in the form of a string "
    " Example return: 0, {}, EnableRobot() -> (None, "") "
    def send_command(self, cmd: str) -> Tuple[Optional[DobotError], str]:
        encoded_cmd = cmd.encode("utf-8")
        self.socket.send(encoded_cmd)
        log.debug(f'The command "{encoded_cmd}" has been sent.')
        return self.__await_reply()

    " This is a quick solution, but may not cover all errors - in which it might panic. "
    def __await_reply(self) -> Tuple[Optional[DobotError], str]:
        data = self.socket.recv(1024)
        response: str = str(data, encoding="utf-8")
        log.debug(f'The return message was "{response}".')
        split_response = response.split(",")
        errorID: int = int(split_response[0].strip())
        return_value: str = split_response[1].strip()
        if errorID == 0:
            return (None, return_value[1:-1])
        else:
            " This will panic if there is no error for the value of errorID, make a pull request. "
            return (DobotError(errorID), return_value[1:-1])
    
    def close(self):
        if self.socket:
            self.socket.close()

    def __del__(self):
        self.close()

class Simulator:
    def __init__(self, fn: URDF) -> None:
        self.chain = Chain.from_urdf_file(fn)

    def compute(self, target_position: list[float]) -> list[float]:
        self.chain.inverse_kinematics(target_position)

        
def clamp(val: int, local_min: int, local_max: int) -> int:
    log.info(f"{val} was clamped to the range {local_min}, {local_max}")
    return max(local_min, min(val, local_max))

     