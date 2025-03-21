from setuptools import setup, find_packages

setup(
    name='deeptmhmm',
    version='1.0.0',
    description='DeepTMHMM: Academic version for transmembrane region prediction',
    author='Sandy Macdonald',
    author_email='sandyjmacdonald@gmail.com',
    packages=find_packages(),
    include_package_data=True,  # Include package data specified in package_data or MANIFEST.in
    install_requires=[
        # Build dependencies
        "wheel",
        "Cython==0.29.37",
        "pkgconfig==1.5.5",
        # Install PyTorch from a direct URL using PEP 508 direct URL specifier
        "torch @ https://download.pytorch.org/whl/cu92/torch-1.5.0%2Bcu92-cp38-cp38-linux_x86_64.whl#sha256=77586f5deca99bf854dce2bce9e533a90dd97694d190b15bd17c170ef493e2b1",
        # Other dependencies (copied from requirements.txt)
        "biopython==1.79",
        "certifi==2022.9.24",
        "charset-normalizer==2.1.1",
        "cycler==0.11.0",
        "fair-esm==0.4.0",
        "fonttools==4.38.0",
        "future==0.18.2",
        "h5py==3.7.0",
        "idna==3.4",
        "kiwisolver==1.4.4",
        "matplotlib==3.5.2",
        "numpy==1.23.4",
        "packaging==21.3",
        "PeptideBuilder==1.1.0",
        "Pillow==9.3.0",
        "pyparsing==3.0.9",
        "python-dateutil==2.8.2",
        "requests==2.28.1",
        "six==1.16.0",
        "tqdm==4.64.1",
        "urllib3==1.26.12"
    ],
    package_data={
        # This includes any .model and .pt files in the deeptmhmm package
        'deeptmhmm': ['*.model', '*.pt']
    },
)