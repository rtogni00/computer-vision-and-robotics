import ev3_dc as ev3
# import time

# robot = ev3.EV3(protocol=ev3.USB, sync_mode="STD")

# print(robot.sensors)


# motor_right = ev3.Motor(ev3.PORT_A, ev3_obj=robot)
# motor_left = ev3.Motor(ev3.PORT_D, ev3_obj=robot)
# color_left= ev3.Color(ev3.PORT_4, ev3_obj=robot)
# color_right= ev3.Color(ev3.PORT_1, ev3_obj=robot)
# print(color_left.color, color_right.color)
# print(color_left.reflected, color_right.reflected)



# speed_l=10
# speed_r=10
# velocity=1
# brake=True

# gyro = ev3.Gyro(ev3.PORT_3, ev3_obj=robot)
# current_angle = gyro.angle

# while gyro.angle < 360:
#     if motor_right.busy:
#          pass
#     else:
#         motor_right.start_move_by(velocity*45, speed=speed_r, brake=brake)
#         motor_left.start_move_by(-velocity*45, speed=speed_l, brake=brake)
    
#     current_angle = gyro.angle


# with ev3.Gyro(ev3.PORT_3, ev3_obj=robot) as gyro:
#     while True:
#         current_angle = gyro.angle
#         print(f"\rThe current angle is {current_angle:4d} °", end='')
#         if current_angle >= 360:
#             print("\n" + "The sensor made a full clockwise turn!")
#             break
#         elif current_angle <= -360:
#             print("\n" + "The sensor made a full counterclockwise turn!")
#             break
#         time.sleep(0.05)




with ev3.TwoWheelVehicle(
    0.028,  # radius_wheel_measured
    0.028,  # tread
    protocol=ev3.USB,
    # sync_mode="STD",
) as my_vehicle:
    my_vehicle.drive_straight(0.20).start(thread=False)
    print(my_vehicle.motor_pos.left)