import os
import sys
from setuptools import setup, find_packages


def get_version(filename):
    import os
    import re

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, filename)) as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = get_version('aiodataloader.py')

tests_require = [
    'pytest>=2.7.3', 'pytest-cov', 'coveralls', 'mock', 'pytest-asyncio'
]

setup(
    name='aiodataloader',
    version=version,
    description='Asyncio DataLoader implementation for Python',
    long_description=open('README.rst').read(),
    url='https://github.com/syrusakbary/aiodataloader',
    download_url='https://github.com/syrusakbary/aiodataloader/releases',
    author='Syrus Akbary',
    author_email='me@syrusakbary.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ],
    keywords='concurrent future deferred aiodataloader',
    py_modules=['aiodataloader'],
    extras_require={
        'test': tests_require,
    },
    tests_require=tests_require, )
