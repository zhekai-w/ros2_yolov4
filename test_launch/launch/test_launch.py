import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='tutorial_pubsub',
            executable='tutorial_talker',
            name='talker'),

        launch_ros.actions.Node(
            package='tutorial_pubsub',
            executable='tutorial_subscriber',
            name='subscriber'),
  ])