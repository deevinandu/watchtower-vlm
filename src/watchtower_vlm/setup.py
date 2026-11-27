from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'watchtower_vlm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deevinandu',
    maintainer_email='deevinandu@gmail.com',
    description='Semantic safety node: local VLM anomaly detection publishing an E-Stop.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'watchtower_node = watchtower_vlm.watchtower_node:main',
            'benchmark = watchtower_vlm.benchmark:main',
        ],
    },
)
