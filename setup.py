import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reid-metric", # Replace with your own username
    version="0.0.1",
    author="Namgyu Ho",
    author_email="itsnamgyu@gmail.com",
    url="https://github.com/itsnamgyu/reid-metric",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)


