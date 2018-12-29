.PHONY: all clean test

profile:
	kernprof -l -v strlearn/tests/test_ensembles.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf doc/_build
	rm -rf doc/auto_examples
	rm -rf doc/generated
	rm -rf doc/modules
	rm -rf examples/.ipynb_checkpoints

docs: clean install
	cd doc && make

test-code:
	py.test strlearn

test-coverage:
	rm -rf coverage .coverage
	py.test --cov=strlearn strlearn

test: clean test-coverage

run: clean
	python workspace.py

code-analysis:
	flake8 strlearn | grep -v __init__
	pylint -E strlearn/ -d E1103,E0611,E1101

upload:
	python setup.py sdist upload -r pypi
	pip install --upgrade stream-learn

install: clean
	python setup.py clean
	python setup.py develop
	#pip uninstall stream-learn --yes
	#python setup.py install --record files.txt
	#cat files.txt | xargs rm -rf
	#rm files.txt
	#python setup.py install --force
