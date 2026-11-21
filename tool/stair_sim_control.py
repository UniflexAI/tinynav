"""Apply the last published command at every physics tick, including resume."""


def apply_drive_command(drive, command, manual=None):
    if manual is not None:
        drive.vx, drive.wz = manual
    else:
        drive.vx = command.linear.x if command is not None else 0.
        drive.wz = command.angular.z if command is not None else 0.
