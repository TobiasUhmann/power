sphinx-apidoc ..\src\ -o source\apidoc\ --implicit-namespaces --separate
make clean
make html
