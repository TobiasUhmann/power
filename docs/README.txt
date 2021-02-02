sphinx-apidoc ..\src\ -o source\apidoc\ --implicit-namespaces --separate --force
make clean
make html
