#!/bin/env/python
#-*- encoding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as README_FILE:
	readme = README_FILE.read()

with open('LICENSE') as LICENSE_FILE:
	license = LICENSE_FILE.read()

setup(
	name='auvsi_suas',
	version='0.1.0',
	description='Software for AUVSI SUAS Clark Aerospace UAV',
	long_description=readme,
	author='Stephen Offer',
	author_email='clarkaerospace1@gmail.com',
	url='https://github.com/clarkaerospace/auvsi_suas',
	license=license,
	packages=find_packages(exclude=('tests','docs'))
)