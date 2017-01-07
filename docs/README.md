Documentation
=============

HTML
----
For publishing, make a clean `hmtl` with *Sphinx* makefile

```
make clean hmtl
```


Publishing in GitHub Pages
--------------------------
Import documentation to `gh-pages` branch with [ghp-import](https://github.com/davisp/ghp-import) package.

From `docs` directory and use command

```
ghp-import -p build/html/
```
