from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description_file = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="survlimepy",
    version="0.0.1",
    description="A python package implementing SurvLIME algorithm",
    long_description=long_description_file,
    long_description_content_type="text/markdown",
    packages=find_packages(where="survlime"),
    install_requires=[
        "numpy",
        "cvxpy",
        "scikit-survival",
        "scikit-learn",
        "pandas",
        "tqdm",
        "seaborn",
        "matplotlib",
    ],
    package_dir={"": "survlimepy"},
    extras_require={"dev": ["pytest"]},
    test_suite="tests",
    package_data={"datasets": ["*.csv"]},
    author="Carlos Hernández Pérez, Cristian Pachón García",
    author_email="crherperez95@gmail.com, cc.pachon@gmail.com",
    keywords="Interpretable Machine Learning, eXplainalbe Artificial Intelligence, Survival Analysis, Machine Learning, Python",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
    ],
    project_urls={
        "Bug Reports": "https://github.com/imatge-upc/SurvLIMEpy/issues",
        "Source": "https://github.com/imatge-upc/SurvLIMEpy/",
    },
)
