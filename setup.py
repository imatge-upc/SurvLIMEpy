from setuptools import setup, find_packages

setup(
    name="survlime",
    version="0.1.8",
    description="Survival adaptation of the LIME algorithm",
    packages=find_packages(),
    install_requires=["numpy", "cvxpy", "scikit-survival", "scikit-learn", "pandas"],
    extras_require={"dev": ["pytest"]},
    test_suite="tests",
    url='https://github.com/imatge-upc/SurvLIME',
    package_data= {
    '': ['datasets/*.csv']
    },
    author='Carlos Hernández Pérez, Cristian Pachón García',
    author_email='crherperez95@gmail.com, cc.pachon@gmail.com'
)
