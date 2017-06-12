import sys
from setuptools import setup, find_packages

version = __import__('aiodataloader').__version__


tests_require = [
    'pytest>=2.7.3', 'pytest-cov', 'coveralls',
    'mock', 'pytest-asyncio'
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
    packages=find_packages(exclude=['tests']),
    extras_require={
        'test': tests_require,
    },
    tests_require=tests_require, )
