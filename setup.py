import setuptools

with open('README.md', 'r') as rmf:
    readme = rmf.read()
    
with open('VERSION', 'r') as verf:
    version = verf.read()
    
setuptools.setup(
    name='holofun',
    version=version,
    author='Will Hendricks',
    author_email='hendricksw@berkeley.edu',
    description='Holography and 2-photon calcium imaging analysis.',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/willyh101/analysis',
    license='MIT',
    packages=setuptools.find_packages()
)