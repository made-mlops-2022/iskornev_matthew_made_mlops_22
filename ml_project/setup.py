from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Matthew Iskornev",
    entry_points={
        "console_scripts": [
            "data_load = src.data.data_load:main",
            "make_synth_data = tests.make_synth_data:main"
        ]
    },
    install_requires=required,
    license="MIT",
)
