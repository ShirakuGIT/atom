from setuptools import find_packages
from setuptools import setup

package_name = 'kdl_parser_py'

setup(
    name=package_name,
    version='3.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Lalancette',
    maintainer_email='clalancette@gmail.com"',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=(
        'The Kinematics and Dynamics Library (KDL)'
        'defines a tree structure to represent the kinematic and'
        'dynamic parameters of a robot mechanism. <tt>kdl_parser_py</tt>'
        'provides Python tools to construct a KDL tree from an XML robot representation in URDF.'
    ),
    license='BSD',
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
