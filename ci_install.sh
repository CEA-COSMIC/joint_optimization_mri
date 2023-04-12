python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/chaithyagr/tf-fastmri-data@master
python -m pip install git+https://github.com/chaithyagr/fastmri-reproducible-benchmark@master
python -m pip install git+https://jopmri:${SPARKLING_TOKEN}@gitlab.com/chaithyagr/CSMRI_sparkling
python -m pip install --upgrade -e .
