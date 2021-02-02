sphinx-apidoc ..\src\ -o source\apidoc\ --implicit-namespaces --separate --force --module-first --no-toc
make clean
make html
