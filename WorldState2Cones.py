#!/usr/bin/env python
import rospy
import rosbag
import csv
from geometry_msgs.msg import Point, Pose

# time of last world state message
msg_time = rospy.Time(1565423386, 883208788)

bag = rosbag.Bag("/home/bedskes/rosbags/fsg_trackdrive.bag")
cones = []
for topic, msg, t in bag.read_messages(topics = ['/world_state'], start_time = msg_time, end_time = msg_time):
    objects = msg.objects
    for object in objects:
        for property in object.properties:
            if property.attribute == "position":
                cones.append(Point(property.pdf.data[1], property.pdf.data[2], 0.29))

with open('fast_raycast/src/test2.csv', mode = 'w') as cone_file:
    cone_writer = csv.writer(cone_file, delimiter=',')
    for cone in cones:
        cone_writer.writerow([cone.x, cone.y, cone.z])
print("GOT CONES")

positions = []
for topic, msg, t in bag.read_messages(topics = ['/tf']):
    for tf in msg.transforms:
        if tf.header.frame_id == "map" and tf.child_frame_id == "base_link":
            pos = Pose()
            pos.position = tf.transform.translation
            pos.position.z = 0
            pos.orientation = tf.transform.rotation
            positions.append(pos)

# output to csv with position, orientation: x, y, z, x, y, z, w
with open('fast_raycast/src/test_positions.csv', mode = 'w') as position_file:
    position_writer = csv.writer(position_file, delimiter=',')
    for pos in positions:
        position_writer.writerow([pos.position.x,
                                  pos.position.y,
                                  pos.position.z,
                                  pos.orientation.x,
                                  pos.orientation.y,
                                  pos.orientation.z,
                                  pos.orientation.w])
print("GOT POSITIONS")
