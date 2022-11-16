from setuptools import setup, find_packages

setup(
    name="survlime",
    version="0.1.0",
    description="Survival adaptation of the LIME algorithm",
    package_dir={"": "."},
    packages=find_packages(),
    install_requires=["numpy", "cvxpy", "sklearn", "pandas"],
    extras_require={"dev": ["pytest"]},
    test_suite="tests",
    url='https://github.com/imatge-upc/SurvLIME',
    include_package_data=True,
    author='Carlos Hernández Pérez, Cristian Pachón García',
    author_email='crherperez95@gmail.com, cc.pachon@gmail.com'
)
