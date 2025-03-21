### DeepTMHMM 1.0 - Academic Version ###

### Installation ###

# Install Python 3.8
sudo apt-get install python3.8 python3.8-dev python3.8-venv libhdf5-dev

# Setup a virtual environment (optional)
python3.8 -m venv ../DeepTMHMM-venv
source ../DeepTMHMM-venv/bin/activate

# Install build dependencies
python3 -m pip install wheel Cython==0.29.37 pkgconfig==1.5.5

# Install PyTorch
python3 -m pip install https://download.pytorch.org/whl/cu92/torch-1.5.0%2Bcu92-cp38-cp38-linux_x86_64.whl#sha256=77586f5deca99bf854dce2bce9e533a90dd97694d190b15bd17c170ef493e2b1

# Install other dependencies
python3 -m pip install -r requirements.txt

# Run tool on sample file
python3 predict.py --fasta sample.fasta --output-dir result1

# The result is now available in result1/


### Support ###
Please see https://dtu.biolib.com/DeepTMHMM/ for additional documentation. For any questions, please contact licensing@biolib.com.

### License ###
This source code in the package is made available under a academic license. You are not permitted to use it for any commercial purpose. To acquire a commercial license, please contact licensing@biolib.com.

The package uses the software dependencies referenced in the above installation instructions. Please ensure that you understand and accept the licenses of those. The package uses an embedded version of pytorch-crf, you can find the license of this in the file at pytorchcrf/LICENSE.txt. This package embeds the ESM1b model, you can find the license of this in the file at ESM-License.txt.

