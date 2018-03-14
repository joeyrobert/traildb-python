# Python bindings for TrailDB

### Quick start

First install the [TrailDB library](https://github.com/traildb/traildb). Then

    $ python setup.py install

For detailed instructions, see [Getting Started guide](http://traildb.io/docs/getting_started/).

### Example

See [TrailDB tutorial](http://traildb.io/docs/tutorial) for more information.

```python

>>> from traildb import TrailDB, TrailDBConstructor

>>> cookie = '12345678123456781234567812345678'
>>> cons = TrailDBConstructor('test.tdb', ['field1', 'field2'])
>>> cons.add(cookie, 123, ['a'])
>>> cons.add(cookie, 124, ['b', 'c'])
>>> tdb = cons.finalize()

>>> for cookie, trail in tdb.trails():
...     for event in trail:
...         print cookie, event

12345678123456781234567812345678 event(time=123L, field1='a', field2='')
12345678123456781234567812345678 event(time=124L, field1='b', field2='c')
```

## For Docker User:

You can pull image from here:

    $ docker pull c3h3/traildb-ipynb

Or, you can build docker image by yourself (please replace "your/repo-name" with whatever you want):

    $ docker build -t your/repo-name .


You can run the docker image with default password (jupyter), and your jupyter notebook will listen on 8080 port:

    $ docker run -p 8080:8888 -it c3h3/traildb-ipynb

Or, you can run the docker image with your password (yourPassword), and your jupyter notebook will listen on 8080 port:

    $ docker run -e PASSWORD=yourPassword -p 8080:8888 -it c3h3/traildb-ipynb

Easily to use [http://localhost:8080](http://localhost:8080) to access your jupyter notebook

#### Documentation

Sphinx documentation is available.

1. Ensure Sphinx is installed.

`pip install sphinx`

2. Generate HTML documentation.

`sphinx-build -b html . _build`

Open `_build/index.html` in a browser.
