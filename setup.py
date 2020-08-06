
from setuptools import find_packages, setup

with open("README.rst", "r") as fh:
    LONG_DESCRIPTION = fh.read()

DISTNAME = 'scikit-clean'
DESCRIPTION = 'A collection of algorithms for detecting and handling label noise'
MAINTAINER = 'Shihab Shahriar Khan'
MAINTAINER_EMAIL = 'redoykhan555@gmail.com'
URL = 'https://github.com/Shihab-Shahriar/scikit-clean'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/Shihab-Shahriar/scikit-clean'
VERSION = "0.1.2"
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']

INSTALL_REQUIRES = [
    'scikit-learn>=0.23',
    'pandas>=1.0',
]

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      include_package_data=True)


