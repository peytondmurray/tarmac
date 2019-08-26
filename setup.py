# Written by Peyton Murray
# @peytondmurray
# peytondmurray.github.io

from setuptools import setup

setup(
    name='tarmac',
    packages=['tarmac'],  # this must be the same as the name above
    version='0.7',
    description='Tools for processing and visualization markov chain samples',
    author='Peyton Murray',
    author_email='peynmurray@gmail.com',
    url='https://github.com/peytondmurray/tarmac',
    # use the URL to the github repo
    license="MIT",
    download_url='https://github.com/peytondmurray/tarmac/archive/0.7.zip',
    keywords=['plotting', 'mcmc', 'markov chain', 'monte carlo', "bayesian"],
    # arbitrary keywords
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=["matplotlib", "numpy"],
    python_requires='>=3',
)
