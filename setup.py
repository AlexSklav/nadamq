#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import versioneer

from setuptools import setup, find_namespace_packages

setup(
    name='nadamq',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Embedded-friendly transport layer, inspired by ZeroMQ',
    keywords='python embedded zeromq transport packet parse',
    author='Christian Fobel',
    author_email='christian@fobel.net',
    url='https://github.com/alexsk/nadamq',
    license='GPL',
    packages= ['nadamq'],
    # packages=find_namespace_packages(include=['nadamq*']),
    python_requires='>=3.6',
    include_package_data=True
)
