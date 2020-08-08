from setuptools import setup

setup(
    name='unik',
    version='0.1.0',
    packages=['unik'],
    url='https://github.com/balbasty/unik',
    license='MIT',
    author='Yael Balbastre',
    author_email='yael.balbastre@gmail.com',
    description='A unified seamless interface for Keras, Tensorflow and Numpy',
    python_requires='>=3.5',
    install_requires=['tensorflow>=2', 'numpy', 'scipy'],
)
