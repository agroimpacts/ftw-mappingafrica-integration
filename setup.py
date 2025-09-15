from setuptools import setup, find_packages

setup(
    name="ftw_ma",
    version="0.0.2",
    author="Lyndon Estes",
    author_email="lestes@clarku.edu",
    url="https://github.com/agroimpacts/ftw-mappingafrica-integration",    
    # package_dir={"": "ftw_ma"},
    packages=find_packages(), #(where="ftw_ma"),
    include_package_data=True,
    install_requires=[
        "ftw-tools==1.4.3",
        "ipykernel",
        "leafmap",
        "localtileserver",
        "jupyterlab",
        "scikit-learn",
        "torchgeo",
        # "pyarrow",  # Uncomment if needed
        "geodatasets",
        "makelabels @ git+https://github.com/agroimpacts/lacunalabels.git",
        "instancemaker @ git+https://github.com/agroimpacts/instancemaker.git",
        "torchgeo @ git+https://github.com/microsoft/torchgeo.git"
    ],
    entry_points={
        "console_scripts": [
            "ftw_ma=ftw_ma.cli:ftw_ma",
        ],
    },
)