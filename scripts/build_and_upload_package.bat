@echo off
cls
python3 setup.py bdist_wheel
twine upload dist/*
pause