from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="wraval",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    data_files=[
        ('config', ['config/settings.toml']),
        ('model_artifacts/code', [
            'src/wraval/model_artifacts/code/inference.py',
            'src/wraval/model_artifacts/code/requirements.txt'
        ])
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "wraval=wraval.main:main",
        ],
    }
)
