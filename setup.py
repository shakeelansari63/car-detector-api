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

subprocess.run("pip3 install {}".format(" ".join(required)),
               shell=True, capture_output=True)
