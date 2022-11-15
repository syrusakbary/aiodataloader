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


version = get_version('aiodataloader/__init__.py')

tests_require = [
    'pytest>=3.6', 'pytest-cov', 'coveralls', 'mock', 'pytest-asyncio'
]

setup(
    name='aiodataloader',
    version=version,
    description='Asyncio DataLoader implementation for Python',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
    ],
    keywords='concurrent future deferred aiodataloader',
    py_modules=['aiodataloader'],
    python_requires='>=3.6',
    extras_require={
        'lint': ['flake8', 'mypy'],
        'test': tests_require,
    },
    install_requires=["typing_extensions>=4.3.0"],
    tests_require=tests_require, )
