import logging as log
from enum import IntEnum
import numpy as np
from dataclasses import dataclass
from strenum import StrEnum


MOVEMENT_PORT = 30003
REALTIME_FEEDBACK_PORT = 30004
DASHBOARD_PORT = 29999


# There could be more, but not documented in:
# https://github.com/Dobot-Arm/TCP-IP-Protocol/blob/master/README-EN.md
class DobotError(IntEnum):
    FAIL_TO_GET = -1
    COMMAND_ERROR = -10000
    PARAMETER_NUM_ERROR = -20000
    WRONG_PARAM_TYPE = -30000
    FIRST_PARAM_INCORRECT = -30001
    SECOND_PARAM_INCORRECT = -30002
    PARAMETER_RANGE_INCORRECT = -40000
    FIRST_PARAM_RANGE = -40001
    SECOND_PARAM_RANGE = -40002

@dataclass
class IOPort:
    mode: int
    distance: int
    index: int
    status: int

    def __post_init__(self, mode: int, distance: int, index: int, status: int):
        self.mode = self.__clamp(mode, 0, 1)
        self.distance = self.__clamp(distance, 0, 100)
        self.index = self.__clamp(index, 1, 24)
        self.status = self.__clamp(status, 0, 1)
    
    def __clamp(self, val: int, local_min: int, local_max: int) -> int:
        log.info(f"{val} was clamped to the range {local_min}, {local_max}")
        return max(local_min, min(val, local_max))

class RobotMode(IntEnum):
    INIT = 1
    BRAKE_OPEN = 2
    RESERVED = 3
    DISABLED = 4
    ENABLE = 5
    BACKDRIVE = 6
    RUNNING = 7
    RECORDING = 8
    ERROR = 9
    PAUSE = 10
    JOG = 11

class JointSelection(StrEnum):
    J1NEG = "j1-"
    J1POS = "j1+"
    J2NEG = "j2-"
    J2POS = "j2+"
    J3NEG = "j3-"
    J3POS = "j3+"
    J4NEG = "j4-"
    J4POS = "j4+"
    J5NEG = "j5-"
    J5POS = "j5+"


class RobotType(IntEnum):
    CR3 = 3
    CR3L = 31
    CR5 = 5
    CR7 = 7
    CR10 = 10
    CR12 = 12
    CR16 = 16
    MG400 = 1
    M1PRO = 2
    NOVA2 = 101
    NOVA5 = 103
    CR3V2 = 113
    CR5V2 = 115
    CR10V2 = 120

class URDF(StrEnum):
    M1PRO = "urdf/m1pro_description.urdf"

import numpy as np

FeedbackType = np.dtype([
    ('len', np.int32),                  # 4 bytes: Size of package
    ('digital_input_bits', np.uint64),  # 8 bytes: DI status
    ('digital_output_bits', np.uint64), # 8 bytes: DO status
    ('robot_mode', np.uint64),          # 8 bytes: Core Robot Mode status
    ('time_stamp', np.uint64),          # 8 bytes: Controller tick clock
    ('time_stamp_reserve_bit', np.uint64),
    ('test_value', np.uint64),
    ('test_value_keep_bit', np.uint64), # 8 bytes: Standard reserved alignment point
    
    ('q_actual', np.float64, (6,)),     # 48 bytes: Joint positions
    ('q_target', np.float64, (6,)),     # 48 bytes: Target joint positions
    ('qd_actual', np.float64, (6,)),    # 48 bytes: Joint velocities
    ('qdd_actual', np.float64, (6,)),   # 48 bytes: Joint accelerations
    ('i_actual', np.float64, (6,)),     # 48 bytes: Actual currents
    ('i_target', np.float64, (6,)),     # 48 bytes: Target currents
    
    # --- CRITICAL FOR POSITION TRACKING (Red Dot) ---
    ('tool_vector_actual', np.float64, (6,)), # X, Y, Z, R (plus padding up to 6 floats)
    ('tool_vector_target', np.float64, (6,)), # Target Cartesian coordinates
    
    ('v_actual', np.float64, (6,)),     # 48 bytes: Cartesian velocity
    ('a_actual', np.float64, (6,)),     # 48 bytes: Cartesian acceleration
    ('tcp_force', np.float64, (6,)),    # 48 bytes: Force vectors
    ('tcp_force_target', np.float64, (6,)),
    ('w_actual', np.float64, (6,)),     # 48 bytes: Target tool weight vectors
    ('w_target', np.float64, (6,)),
    
    # Remainder 24 bytes up to 432 bytes are state bytes / padding
    ('status_bytes', np.uint8, (24,))
])