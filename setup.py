import sys
from setuptools import setup
import subprocess

with open('requirements.txt', 'r') as req:
    required = list(
        map(lambda x: x.strip().strip('\n'),
            filter(lambda x: not x.startswith('#'),
                   req.readlines()
                   )
            )
    )

print("Installing required packages:", *required, sep=' ')

setup(
    # basic package data
    name='Car Detector API',
    version='0.1',
    author='Shakeel Ansari',
    author_email='shakeel.ansari@gmail.com',
    license='',
    url='https://github.com/shakeelansari63/car-detector-api',
    install_requires=required,
)

print("Running command: pip3 install {}".format(" ".join(required)))

for pkg in required:
    subprocess.run("pip3 install {}".format(pkg),
                   shell=True, capture_output=True)
