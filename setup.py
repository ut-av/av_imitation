from setuptools import setup
import os
from glob import glob

package_name = 'av_imitation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name,
        f'{package_name}.webapp',
        f'{package_name}.scripts',
        f'{package_name}.src',
        f'{package_name}.src.models',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'webapp', 'templates'), glob('av_imitation/webapp/templates/*')),
        (os.path.join('share', package_name, 'webapp', 'static', 'js'), glob('av_imitation/webapp/static/js/*')),
    ],
    install_requires=['setuptools', 'Flask', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Tools for imitation learning from ROS 2 bag files',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start_webapp = av_imitation.webapp.app:main',
            'process_bag = av_imitation.scripts.process_bag:main',
            'export_onnx = av_imitation.scripts.export_onnx:main',
            'inference = av_imitation.scripts.inference:main',
            'play = av_imitation.src.play:main',
        ],
    },
)
