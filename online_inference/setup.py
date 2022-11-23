from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="microservice",
    packages=find_packages(),
    version="0.1.0",
    description="service for making online prediction of our model",
    author="Matthew Iskornev",
    install_requires=required,
)
