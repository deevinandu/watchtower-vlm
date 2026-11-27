"""Launch the Watchtower VLM semantic safety node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('watchtower_vlm'), 'config', 'watchtower.yaml')

    overrides = {
        'image_topic': '/camera/image_raw',
        # YAML list, e.g. prompts:="['Is liquid spilling?', 'Has something fallen?']"
        'prompts': "['Is there a liquid spill, a dropped object, or a collision?']",
        'confidence_threshold': '0.5',
        'model_path': 'vikhyatk/moondream2',
        'device': 'auto',
        'backend': 'transformers',
        'gguf_file': "''",  # quoted so launch_ros parses it as an empty string
        'n_gpu_layers': '-1',
    }
    args = [DeclareLaunchArgument('config', default_value=default_config,
                                  description='Parameter YAML file.')]
    args += [DeclareLaunchArgument(name, default_value=value) for name, value in overrides.items()]

    node = Node(
        package='watchtower_vlm',
        executable='watchtower_node',
        name='watchtower_vlm',
        output='screen',
        parameters=[
            LaunchConfiguration('config'),
            {name: LaunchConfiguration(name) for name in overrides},
        ],
    )
    return LaunchDescription(args + [node])
