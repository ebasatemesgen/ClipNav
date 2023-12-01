Step 1: Add this to model.sdf file 


    <link name="image">
      <inertial>
        <pose>-0.069 0 0.104 0 0 0</pose>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.000</ixy>
          <ixz>0.000</ixz>
          <iyy>0.001</iyy>
          <iyz>0.000</iyz>
          <izz>0.001</izz>
        </inertia>
        <mass>0.035</mass>
      </inertial>

      <collision name="image_collision">
        <pose>-0.069 0 0.104 0 0 0</pose>
        <geometry>
          <box>
            <size>0.008 0.130 0.022</size>
          </box>
        </geometry>
      </collision>

      <sensor name="intel_realsense_r200" type="camera">
        <always_on>1</always_on>
        <visualize>0</visualize>
        <update_rate>30</update_rate>
        <pose>0.064 -0.047 0.107 0 0 0</pose>
        <camera>
          <horizontal_fov>1.02974</horizontal_fov>
          <image>
            <width>1920</width>
            <height>1080</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>

Step 2: Add this to model-1_4.sdf file

  <link name="image">
      <inertial>
        <pose>-0.069 0 0.104 0 0 0</pose>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0.000</ixy>
          <ixz>0.000</ixz>
          <iyy>0.001</iyy>
          <iyz>0.000</iyz>
          <izz>0.001</izz>
        </inertia>
        <mass>0.035</mass>
      </inertial>

      <collision name="image_collision">
        <pose>-0.069 0 0.104 0 0 0</pose>
        <geometry>
          <box>
            <size>0.008 0.130 0.022</size>
          </box>
        </geometry>
      </collision>

      <sensor name="intel_realsense_r200" type="camera">
        <always_on>1</always_on>
        <visualize>0</visualize>
        <update_rate>30</update_rate>
        <pose>0.064 -0.047 0.107 0 0 0</pose>
        <camera>
          <horizontal_fov>1.02974</horizontal_fov>
          <image>
            <width>1920</width>
            <height>1080</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
      </sensor>
    </link>

Step 3: modify your launch file to include the /opt/ros/noetic/share/turtlebot3_description/urdf


  <joint name="camera_joint" type="fixed">
    <origin xyz="0.073 -0.011 0.084" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="camera_link"/>
  </joint>

  <link name="camera_link">
    <collision>
      <origin xyz="0.005 0.011 0.013" rpy="0 0 0"/>
      <geometry>
        <box size="0.015 0.030 0.027"/>
      </geometry>
    </collision>
  </link>

  <joint name="camera_rgb_joint" type="fixed">
    <origin xyz="0.003 0.011 0.009" rpy="0 0 0"/>
    <parent link="camera_link"/>
    <child link="camera_rgb_frame"/>
  </joint>
  <link name="camera_rgb_frame"/>

  <joint name="camera_rgb_optical_joint" type="fixed">
    <origin xyz="0 0 0" rpy="-1.57 0 -1.57"/>
    <parent link="camera_rgb_frame"/>
    <child link="camera_rgb_optical_frame"/>
  </joint>
  <link name="camera_rgb_optical_frame"/>



/opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro


<!--link : https://www.raspberrypi.org/documentation/hardware/camera/-->
  <gazebo reference="camera_rgb_frame">
    <sensor type="camera" name="Pi Camera">
      <always_on>true</always_on>
      <visualize>$(arg camera_visual)</visualize>
      <camera>
          <horizontal_fov>1.085595</horizontal_fov>
          <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
          </image>
          <clip>
              <near>0.03</near>
              <far>100</far>
          </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>30.0</updateRate>
        <cameraName>camera</cameraName>
        <frameName>camera_rgb_optical_frame</frameName>
        <imageTopicName>rgb/image_raw</imageTopicName>
        <cameraInfoTopicName>rgb/camera_info</cameraInfoTopicName>
        <hackBaseline>0.07</hackBaseline>
        <distortionK1>0.0</distortionK1>
        <distortionK2>0.0</distortionK2>
        <distortionK3>0.0</distortionK3>
        <distortionT1>0.0</distortionT1>
        <distortionT2>0.0</distortionT2>
      </plugin>
    </sensor>
  </gazebo>
