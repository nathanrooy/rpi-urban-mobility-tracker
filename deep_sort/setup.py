from setuptools import setup

import os.path
# Change to the directory of this file before the build
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Build setup
setup(
    name='deep_sort',
    version='1.1.1',
    packages=['deep_sort', 'deep_sort_tools'],
    license='GNU General Public License v3 (GPLv3)',
    author='Nicolai Wojke and Alex Bewley',
    url='https://github.com/mk-michal/deep_sort',
    description='Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)',
     setup_requires=[
        'setuptools>=18.0'
    ],
    package_data={
        'deep_sort': [
            'resources/*',
        ]
    },
    install_requires=[],  # requirements.txt is included
    include_package_data=True,
)


