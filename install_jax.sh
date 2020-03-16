export PYTHON_VERSION=cp37
export CUDA_VERSION=cuda100
export PLATFORM=linux_x86_64
export BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.40-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax
