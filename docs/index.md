# wring

A tool to compress data files.  

Wring can compress csv files into parquet to save substantial disk space and
improve data read speed.  It can also convert OMX files to use blosc2:zstd 
data compression, and reduce floating point arrays from 64 bit to 32 bit.

Wring is available as [`wring`](https://pypi.org/project/wring/) on PyPI:

```shell
pip install wring[omx]
```

If you don't need OMX support, you can install without the optional dependencies:

```shell
pip install wring
```

Wring is primarily a command line tool.  For example, to crawl the current
directory and compress csv files into parquet, use:

```shell
wring csv
```

Or to convert OMX files to use blosc2:zstd data compression and reduce floating
point arrays from 64 bit to 32 bit, use:

```shell
wring omx
```

For an overview of options, see [_Usage_](usage.md).