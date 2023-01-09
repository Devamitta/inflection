from setuptools import setup

setup(
    name='inflection_generator',
    version='0.0.1',
    author='Devamitta bhikkhu',
    author_email='devamitta@sasanarakkha.org',
    python_requires='>=3.8',
    url='https://github.com/Devamitta/inflection-generator-dps',
    license=None,
    description='Generate inflections for Pāli dictionaries',
    install_requires=(
        'aksharamukha~=2.0',
        'openpyxl~=3.0',
        'pandas-ods-reader~=0.1',
        'pandas~=1.0',
        'rich~=12.0',
    ),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'inflection-generator=inflection_generator.cli:main',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
        'Operating System :: OS Independent',
    ],
    py_modules=['inflection_generator'])
