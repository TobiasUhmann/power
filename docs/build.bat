sphinx-apidoc ..\src\ -o source\apidoc\ ^
    --force ^
    --implicit-namespaces ^
    --module-first ^
    --no-toc ^
    --separate ^
    --templatedir apidoc_template\

rmdir /S /Q build\
make html
