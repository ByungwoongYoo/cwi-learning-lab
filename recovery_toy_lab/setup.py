from setuptools import setup,find_packages
setup(name='recovery-toy-lab',version='0.1.0',description='Bounded CPU checkpoint-recovery reproduction demo',packages=find_packages(),python_requires='>=3.10',install_requires=['numpy>=1.26','torch>=2.6'],entry_points={'console_scripts':['recovery-toy=recovery_tool.__main__:main']})
