from setuptools import setup, find_packages

setup(
    name='embed_with_vertex',
    version='0.1.0',
    description='A project to embed with vertex',
    author='Moti Dabastani ',
    author_email='dabastany@gmail.com',
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform==1.40.0",
        "sentence-transformers==2.3.0",
        "transformers==4.36.1",
        "polars==0.20.5",
        "transformers==4.36.1",
        "datasets==2.16.0"
        ],
    entry_points={
        'console_scripts': [
            'embed_with_vertex=embed_with_vertex.vertex_encoder:main',
        ],
    },
)