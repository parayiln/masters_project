# masters_project
Follow the leader controller

run_follow_the_leader.py is the main code.

The follow the leader controller is divided into 4 parts

1. pca.py
  - Contains image processing and PCA algorithm
  - Bezier curves
 
2. controller.py
  - contains the controller code (velocity and position controller for both joint and end effector input controller)

3. sensor_rgb.py
  - contains all the sensor related codes
  - image process for leader centering and following after PCA.

4. robot_bullet.py
  - set up the robot evnironment, joints info and kenimatics. 
