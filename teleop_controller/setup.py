from setuptools import find_packages, setup

package_name = 'teleop_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shiraku', # You can change this
    maintainer_email='shiraku@todo.todo', # You can change this
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tf_listener = teleop_controller.tf_listener_node:main',
            'param_checker = teleop_controller.param_checker:main',
            'hello_node = teleop_controller.hello_node:main',
            'neck_tracker = teleop_controller.neck_tracker:main',
        ],
    },
)