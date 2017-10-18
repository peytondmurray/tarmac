#Written by Peyton Murray
# @peytondmurray
# peytondmurray.github.io

from distutils.core import setup
setup(
	name = 'tarmac',
	packages = ['tarmac'], # this must be the same as the name above
	version = '0.1',
	description = 'Tools for processing and visualisation markov chain samples',
	author = 'Peyton Murray',
	author_email = 'peynmurray@gmail.com',
	url = 'https://github.com/peytondmurray/tarmac', # use the URL to the github repo
	download_url = 'https://github.com/peterldowns/mypackage/archive/0.1.tar.gz', # I'll explain this in a second
	keywords = ['plotting', 'mcmc', 'markov chain', 'monte carlo', "bayesian"], # arbitrary keywords
	classifiers = [],
)