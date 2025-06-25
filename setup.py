from setuptools import setup, find_packages

setup(
    name="arkimede",
    version="0.0.1",
    url="https://github.com/raffaelecheula/arkimede.git",
    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",
    description="Automatic reactions kinetic mechanism design.",
    license="GPL-3.0",
    packages=["arkimede"],
    package_dir={"arkimede": "arkimede"},
    install_requires=[],
    python_requires=">=3.5, <4",
)
