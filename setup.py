from setuptools import find_packages, setup

setup(
    name="pakistan-ihhn",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "Click",
    ],
    entry_points="""
        [console_scripts]
        pakistan-ihhn=main:cli
    """,
    description="""
        A project to improject triaging in the emergency department
         at the Indus Valley Hospital in Pakistan
    """,
    author="DSSG",
    license="",
)
