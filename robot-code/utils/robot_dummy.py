from types import SimpleNamespace


class DummyColour():
    def __init__(self) -> None:
        self.intensity = 0.0
        self.ambient = 0.0


class DummyVehicle():
    def __init__(self) -> None:
        self.motor_pos = SimpleNamespace()
        self.motor_pos.left = 0
        self.motor_pos.right = 0
        pass

    def move(self, speed, turn):
        pass

    def stop(self, brake):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass