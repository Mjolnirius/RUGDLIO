from setuptools import setup, find_packages

setup(
    name = "FeatureLIOM",
    version = "0.0.1",
    packages = find_packages(),
    package_data = {
        'FeatureLIOM': ['config/*.yaml', 'checkpoints/*']
    },
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'Zihao Dong',
    maintainer_email = 'dong.zih@northeastern.edu',
    description = "Dense Learned Compression for SLAM",
    license = "MIT",
    entry_points = {
        'console_scripts': [
            'extract = keypoint_node.keypoint_node:main',           # Input/Output --> ROS subscriber/publisher
            'extract_batch = keypoint_node.batch_runner:main'       # Input/Output --> Folders
        ],
    }
)