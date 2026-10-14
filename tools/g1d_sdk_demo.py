"""Small arm/torso/gripper movement through Unitree SDK on simulation DDS domain 1."""

import argparse
import math
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.utils.crc import CRC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default=None, help="Match the simulator's --dds-interface")
    parser.add_argument("--arm-topic", choices=["rt/lowcmd", "rt/arm_sdk"], default="rt/lowcmd")
    parser.add_argument("--controller-alignment", action="store_true", help="Send the aligned per-joint gains")
    parser.add_argument("--check", action="store_true", help="Assert motion and gripper feedback for an integration check")
    args = parser.parse_args()
    ChannelFactoryInitialize(1, args.interface)
    states = {}
    lock = threading.Lock()

    def receive(name, msg):
        with lock:
            states[name] = msg

    subscribers, publishers = [], []
    try:
        for name, topic, datatype in [("robot", "rt/lowstate", LowState_),
                                      ("left", "rt/dex1/left/state", MotorStates_),
                                      ("right", "rt/dex1/right/state", MotorStates_)]:
            subscriber = ChannelSubscriber(topic, datatype)
            subscriber.Init(lambda msg, name=name: receive(name, msg), 1)
            subscribers.append(subscriber)
        for topic, datatype in [(args.arm_topic, LowCmd_), ("rt/dex1/left/cmd", MotorCmds_), ("rt/dex1/right/cmd", MotorCmds_)]:
            publisher = ChannelPublisher(topic, datatype)
            publisher.Init()
            publishers.append(publisher)
        deadline = time.monotonic() + 15
        while len(states) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(states) < 3:
            raise RuntimeError("No complete simulation feedback. Start run_g1d_sort.py --sdk with the same DDS interface.")
        with lock:
            initial = [m.q for m in states["robot"].motor_state]
            initial_grip = [states[side].states[0].q for side in ("left", "right")]
            mode = states["robot"].mode_machine
        crc = CRC()
        command = unitree_hg_msg_dds__LowCmd_()
        command.mode_machine = mode
        command.mode_pr = 0
        for i in [12, *range(15, 29)]:
            command.motor_cmd[i].mode = 1
            command.motor_cmd[i].q = initial[i]
            if args.controller_alignment:
                wrist = i in (19, 20, 21, 26, 27, 28)
                command.motor_cmd[i].kp = 300.0 if i == 12 else (40.0 if wrist else 80.0)
                command.motor_cmd[i].kd = 1.5 if wrist else 3.0
            else:
                command.motor_cmd[i].kp = 80.0
                command.motor_cmd[i].kd = 3.0
        if args.arm_topic == "rt/arm_sdk":
            command.motor_cmd[29].q = 1.0
        grippers = [MotorCmds_(), MotorCmds_()]
        for msg in grippers:
            msg.cmds.append(unitree_go_msg_dds__MotorCmd_())
            msg.cmds[0].kp, msg.cmds[0].kd = 5.0, 0.05

        def move(targets, grips, seconds=3.0):
            for index, q in targets.items():
                command.motor_cmd[index].q = q
            for msg, q in zip(grippers, grips):
                msg.cmds[0].q = q
            command.crc = crc.Crc(command)
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                for publisher, msg in zip(publishers, [command, *grippers]):
                    publisher.Write(msg)
                time.sleep(1 / 50)

        print("Moving wrists and torso slightly; closing left gripper, then right.", flush=True)
        move({21: initial[21] + 0.2, 28: initial[28] - 0.2, 12: initial[12] + 0.12}, [0.0, 5.4])
        if args.check:
            with lock:
                assert abs(states["robot"].motor_state[21].q - (initial[21] + 0.2)) < 0.10
                assert abs(states["robot"].motor_state[28].q - (initial[28] - 0.2)) < 0.10
                assert abs(states["robot"].motor_state[12].q - (initial[12] + 0.12)) < 0.05
                assert states["left"].states[0].q < 0.5
                assert states["right"].states[0].q > 4.9
                assert crc.Crc(states["robot"]) == states["robot"].crc
        move({}, [5.4, 0.0])
        if args.check:
            with lock:
                assert states["left"].states[0].q > 4.9
                assert states["right"].states[0].q < 0.5
        move({i: initial[i] for i in [12, *range(15, 29)]}, initial_grip)
        print(f"SDK_DEMO_PASSED: {args.arm_topic}, torso yaw and both Dex1 grippers; returned to starting pose.", flush=True)
    finally:
        for channel in subscribers + publishers:
            channel.Close()


if __name__ == "__main__":
    main()
