from setuptools import setup, find_packages

setup(
    name='survLime',
    version='0.0.1',
    description='Survival adaptation of the LIME algorithm',
    package_dir={'': '.'},
    packages=find_packages(),
    install_requires=['numpy', 'cvxpy', 'sklearn', 'pandas'],
    extras_require={'dev': ['pytest']},
    test_suite="tests",
    include_package_data=True,
)
