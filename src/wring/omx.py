from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import console, print
from typing_extensions import Annotated

try:
    import tables as _tb
except ImportError:
    _tb = None
    File = object
else:
    File = _tb.file.File

from ._app import app


def truncate_path_for_display(p: str | Path, maxlen=30):
    """
    Truncate a path if it is too long to display neatly.

    Do not use the truncated path for actual file operations, it won't work.

    Parameters
    ----------
    p : str or pathlib.Path
    maxlen : int
            Maximum number of character to return.  This is a suggestion, as
            the basename itself is never truncated.
    """
    original = Path(p)

    def _trim(x):
        return str(Path("⋯", *original.parts[x:]))  # unicode equivalent '\u22EF'

    if len(str(original)) <= maxlen:
        return str(original)
    original_parts_len = len(original.parts)
    trimmer = 1
    while len(_trim(trimmer)) > maxlen:
        trimmer += 1
        if trimmer >= original_parts_len:
            return _trim(original_parts_len - 1)
    return _trim(trimmer)


def max_len(arg, at_least=0):
    if len(arg) == 0:
        return at_least
    try:
        return max(at_least, *map(len, arg))
    except Exception:
        try:
            return max(at_least, *map(len, *arg))
        except Exception:
            print("ERR max_len")
            print("arg=", arg)
            print("type(arg)=", type(arg))
            raise


class OMX_Error(Exception):
    pass


class OMXBadFormat(OMX_Error):
    pass


class OMXIncompatibleShape(OMXBadFormat):
    pass


class OMXNonUniqueLookup(OMX_Error):
    pass


class OMX(File):
    """A subclass of the :class:`tables.File` class, adding an interface for openmatrix files.

    As suggested in the openmatrix documentation, the default when creating an OMX file
    is to use zlib compression level 1, although this can be overridden.
    """

    def __repr__(self):
        s = f"<OMX> {truncate_path_for_display(self.filename)}"
        s += f"\n |  shape:{self.shape}"
        if len(self.data._v_children):
            s += "\n |  data:"
        just = max_len(self.data._v_children.keys())
        for i in sorted(self.data._v_children.keys()):
            s += "\n |    {0:{2}s} ({1})".format(
                i, self.data._v_children[i].dtype, just
            )
        if len(self.lookup._v_children):
            s += "\n |  lookup:"
        just = max_len(self.lookup._v_children.keys())
        for i in sorted(self.lookup._v_children.keys()):
            s += "\n |    {0:{3}s} ({1} {2})".format(
                i,
                self.lookup._v_children[i].shape[0],
                self.lookup._v_children[i].dtype,
                just,
            )
        return s

    def __init__(self, filename, mode="r", complevel=3, complib="blosc2:zstd", **kwarg):
        if _tb is None:
            raise ImportError("pytables is not installed")
        if "filters" in kwarg:
            super().__init__(filename, mode=mode, **kwarg)
        else:
            super().__init__(
                filename,
                mode=mode,
                filters=_tb.Filters(complib=complib, complevel=complevel),
                **kwarg,
            )
        try:
            self.data = self._get_or_create_path("/data", True)
        except _tb.exceptions.FileModeError:
            raise OMXBadFormat(
                "the '/data' node does not exist and cannot be created"
            ) from None
        try:
            self.lookup = self._get_or_create_path("/lookup", True)
        except _tb.exceptions.FileModeError:
            raise OMXBadFormat(
                "the '/lookup' node does not exist and cannot be created"
            ) from None
        if "OMX_VERSION" not in self.root._v_attrs:
            try:
                self.root._v_attrs.OMX_VERSION = b"0.2"
            except _tb.exceptions.FileModeError:
                raise OMXBadFormat(
                    "the root OMX_VERSION attribute does not exist and cannot be"
                    " created"
                ) from None
        if "SHAPE" not in self.root._v_attrs:
            try:
                self.root._v_attrs.SHAPE = np.zeros(2, dtype=int)
            except _tb.exceptions.FileModeError:
                if mode != "r":
                    raise OMXBadFormat(
                        "the root SHAPE attribute does not exist and cannot be created"
                    ) from None

    @property
    def version(self):
        """The open matrix format version for this file."""
        if "OMX_VERSION" not in self.root._v_attrs:
            raise OMXBadFormat("the root OMX_VERSION attribute does not exist")
        return self.root._v_attrs.OMX_VERSION

    def change_mode(self, mode, **kwarg):
        """
        Change the file mode of the underlying HDF5 file.

        Can be used to change from read-only to read-write.

        Parameters
        ----------
        mode : {'r','a'}
            The file mode to set. Note that the 'w' mode, which would
            open a blank HDF5 overwriting an existing file, is not allowed
            here.

        Returns
        -------
        self
        """
        try:
            if mode == self.mode:
                return
            if mode == "w":
                raise TypeError("cannot change_mode to w, close the file and delete it")
        except RecursionError:
            pass  # the file was probably closed elsewhere, reopen it
        filename = self.filename
        self.close()
        self.__init__(filename, mode, **kwarg)
        return self

    def keys(self, kind="all"):
        """
        Get the names of data and/or lookup stored in this file.

        Parameters
        ----------
        kind : {'all', 'data', 'lookup'}

        Returns
        -------
        list
        """
        if kind == "data":
            return list(self.data._v_children.keys())
        elif kind == "lookup":
            return list(self.lookup._v_children.keys())
        else:
            return list(self.data._v_children.keys()) + list(
                self.lookup._v_children.keys()
            )

    @property
    def shape(self):
        """The shape of the OMX file.

        As required by the standard, all OMX files must have a two-dimensional
        shape. This attribute accesses or alters that shape. Note that attempting
        to change the shape of an existing file that already has data tables that
        would be incompatible with the new shape will raise an OMXIncompatibleShape
        exception.
        """
        sh = self.root._v_attrs.SHAPE[:]
        proposal = (sh[0], sh[1])
        if proposal == (0, 0) and self.data._v_nchildren > 0:
            first_child_name = next(iter(self.data._v_children.keys()))
            proposal = self.data._v_children[first_child_name].shape
            shp = np.empty(2, dtype=int)
            shp[0] = proposal[0]
            shp[1] = proposal[1]
            self.root._v_attrs.SHAPE = shp
        return proposal

    @shape.setter
    def shape(self, x):
        if self.data._v_nchildren > 0:
            if x[0] != self.shape[0] and x[1] != self.shape[1]:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you want to set {x!s}"
                )
        if self.data._v_nchildren == 0:
            shp = np.empty(2, dtype=int)
            shp[0] = x[0]
            shp[1] = x[1]
            self.root._v_attrs.SHAPE = shp

    def add_blank_lookup(
        self, name, atom=None, shape=None, complevel=3, complib="blosc2:zstd", **kwargs
    ):
        """
        Add a blank lookup to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        atom : tables.Atom, optional
            The atomic data type for the new matrix. If not given, Float64 is assumed.
        shape : int, optional
            The length of the lookup to add. Must match one of the dimensions of an existing
            shape.  If there is no existing shape, the shape will be initialized as a square
            with this size.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.

        Returns
        -------
        tables.CArray

        Note
        ----
        This method allows you to add a blank matrix of 3 or more dimemsions,
        only the first 2 must match the OMX.  Adding a matrix of more than
        two dimensions may violate compatability with other open matrix tools,
        do so at your own risk.
        """
        if name in self.lookup._v_children:
            return self.lookup._v_children[name]
        if atom is None:
            atom = _tb.Float64Atom()
        if self.shape == (0, 0) and shape is None:
            raise OMXBadFormat("must set a nonzero shape first or give a shape")
        if shape is not None:
            if shape != self.shape[0] and shape != self.shape[1]:
                if self.shape[0] == 0 and self.shape[1] == 0:
                    self.shape = (shape, shape)
                else:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} but you want to set"
                        f" {shape!s}"
                    )
        else:
            if self.shape[0] != self.shape[1]:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you did not pick one"
                )
            shape = self.shape[0]
        return self.create_carray(
            self.lookup,
            name,
            atom=atom,
            shape=np.atleast_1d(shape),
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_blank_matrix(
        self, name, atom=None, shape=None, complevel=3, complib="blosc2:zstd", **kwargs
    ):
        """
        Add a blank matrix to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        atom : tables.Atom, optional
            The atomic data type for the new matrix. If not given, Float64 is assumed.
        shape : tuple, optional
            The shape of the matrix to add.  The first two dimensions must match the
            shape of any existing matrices; giving a shape is mostly used on initialization.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.

        Returns
        -------
        tables.CArray

        Note
        ----
        This method allows you to add a blank matrix of 3 or more dimemsions,
        only the first 2 must match the OMX.  Adding a matrix of more than
        two dimensions may violate compatability with other open matrix tools,
        do so at your own risk.
        """
        if name in self.data._v_children:
            return self.data._v_children[name]
        if atom is None:
            atom = _tb.Float64Atom()
        if self.shape == (0, 0):
            raise OMXBadFormat("must set a nonzero shape first")
        if shape is not None:
            if shape[:2] != self.shape:
                raise OMXIncompatibleShape(
                    f"this omx has shape {self.shape!s} but you want to set"
                    f" {shape[:2]!s}"
                )
        else:
            shape = self.shape
        return self.create_carray(
            self.data,
            name,
            atom=atom,
            shape=shape,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_matrix(
        self,
        name,
        obj,
        *,
        overwrite=False,
        complevel=3,
        complib="blosc2:zstd",
        ignore_shape=False,
        **kwargs,
    ):
        """
        Add a matrix to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        obj : array-like
            The data for the new matrix.
        overwrite : bool, default False
            Whether to overwrite an existing matrix with the same name.
        complevel : int, default 1
            Compression level.
        complib : str, default 'zlib'
            Compression library.
        ignore_shape : bool, default False
            Whether to ignore all checks on the shape of the matrix being added.
            Setting this to True allows adding a matrix with a different shape
            than any other existing matrices, which may result in an invalid OMX
            file.

        Returns
        -------
        tables.CArray
        """
        obj = np.asanyarray(obj)
        if not ignore_shape:
            if len(obj.shape) != 2:
                raise OMXIncompatibleShape(
                    "all omx arrays must have 2 dimensional shape"
                )
            if self.data._v_nchildren > 0:
                if obj.shape != self.shape:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} but you want to add"
                        f" {obj.shape!s}"
                    )
        if self.data._v_nchildren == 0:
            shp = np.empty(2, dtype=int)
            shp[0] = obj.shape[0]
            shp[1] = obj.shape[1]
            self.root._v_attrs.SHAPE = shp
        if name in self.data._v_children and not overwrite:
            raise TypeError(f"{name} exists")
        if name in self.data._v_children:
            self.remove_node(self.data, name)
        return self.create_carray(
            self.data,
            name,
            obj=obj,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def add_lookup(
        self,
        name,
        obj,
        *,
        overwrite=False,
        complevel=3,
        complib="blosc2:zstd",
        ignore_shape=False,
        **kwargs,
    ):
        """
        Add a lookup mapping to the OMX file.

        Parameters
        ----------
        name : str
            The name of the matrix to add.
        obj : array-like
            The data for the new matrix.
        overwrite : bool, default False
            Whether to overwrite an existing matrix with the same name.
        complevel : int, default 3
            Compression level.
        complib : str, default 'zlib'
            Compression library.
        ignore_shape : bool, default False
            Whether to ignore all checks on the shape of the lookup being added.
            Setting this to True allows adding a lookup with a different length
            than either of the two shape dimensions, which may result in an invalid
            OMX file.

        Returns
        -------
        tables.CArray
        """
        obj = np.asanyarray(obj)
        if not ignore_shape:
            if len(obj.shape) != 1:
                raise OMXIncompatibleShape(
                    "all omx lookups must have 1 dimensional shape"
                )
            if self.data._v_nchildren > 0:
                if obj.shape[0] not in self.shape:
                    raise OMXIncompatibleShape(
                        f"this omx has shape {self.shape!s} "
                        f"but you want to add a lookup with {obj.shape!s}"
                    )
        if self.data._v_nchildren == 0 and self.shape == (0, 0):
            raise OMXIncompatibleShape(
                "don't add lookup to omx with no data and no shape"
            )
        if name in self.lookup._v_children and not overwrite:
            raise TypeError(f"{name} exists")
        if name in self.lookup._v_children:
            self.remove_node(self.lookup, name)
        return self.create_carray(
            self.lookup,
            name,
            obj=obj,
            filters=_tb.Filters(complib=complib, complevel=complevel),
            **kwargs,
        )

    def change_all_atoms_of_type(self, oldatom, newatom):
        for name in self.data._v_children:
            if self.data._v_children[name].dtype == oldatom:
                print(f"changing matrix {name} from {oldatom} to {newatom}")
                self.change_atom_type(name, newatom, matrix=True, lookup=False)
        for name in self.lookup._v_children:
            if self.lookup._v_children[name].dtype == oldatom:
                print(f"changing lookup {name} from {oldatom} to {newatom}")
                self.change_atom_type(name, newatom, matrix=False, lookup=True)

    def change_atom_type(
        self, name, atom, matrix=True, lookup=True, require_smaller=True
    ):
        if isinstance(atom, np.dtype):
            atom = _tb.Atom.from_dtype(atom)
        elif not isinstance(atom, _tb.atom.Atom):
            atom = _tb.Atom.from_type(atom)
        if matrix:
            if name in self.data._v_children:
                orig = self.data._v_children[name]
                neww = self.add_blank_matrix(name + "_temp_atom", atom=atom)
                for i in range(self.shape[0]):
                    neww[i] = orig[i]
                if require_smaller and neww.size_on_disk >= orig.size_on_disk:
                    warnings.warn(
                        f"abort change_atom_type on {name}, {neww.size_on_disk} >"
                        f" {orig.size_on_disk}",
                        stacklevel=2,
                    )
                    neww._f_remove()
                else:
                    neww._f_rename(name, overwrite=True)
        if lookup:
            if name in self.lookup._v_children:
                orig = self.lookup._v_children[name]
                neww = self.add_blank_lookup(name + "_temp_atom", atom=atom)
                for i in range(self.shape[0]):
                    neww[i] = orig[i]
                if require_smaller and neww.size_on_disk >= orig.size_on_disk:
                    warnings.warn(
                        f"abort change_atom_type on {name}, {neww.size_on_disk} >"
                        f" {orig.size_on_disk}",
                        stacklevel=2,
                    )
                    neww._f_remove()
                else:
                    neww._f_rename(name, overwrite=True)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.data._v_children:
                if key in self.lookup._v_children:
                    raise KeyError(f"key {key} found in both data and lookup")
                else:
                    return self.data._v_children[key]
            if key in self.lookup._v_children:
                return self.lookup._v_children[key]
            raise KeyError(f"matrix named {key} not found")
        raise TypeError("OMX matrix access must be by name (str)")

    def __setitem__(self, key, value):
        try:
            value_shape = value.shape
        except Exception as err:
            raise TypeError("value must array-like with one or two dimensions") from err
        if len(value_shape) == 1:
            if value_shape[0] == self.shape[0] or value_shape[0] == self.shape[1]:
                self.add_lookup(key, np.asarray(value))
            else:
                raise OMXIncompatibleShape(
                    f"cannot add vector[{value_shape[0]}] to OMX with shape"
                    f" {self.shape}"
                )
        elif len(value_shape) == 2:
            if value_shape[0] == self.shape[0] and value_shape[1] == self.shape[1]:
                self.add_matrix(key, np.asarray(value))
            else:
                raise OMXIncompatibleShape(
                    f"cannot add matrix[{value_shape}] to OMX with shape {self.shape}"
                )
        else:
            raise OMXIncompatibleShape(
                f"cannot add matrix[{value_shape}] which has more than 3 dimnensions"
                " to OMX"
            )

    def __getattr__(self, key):
        if key in self.data._v_children:
            if key not in self.lookup._v_children:
                return self.data._v_children[key]
            else:
                raise AttributeError(f"key {key} found in both data and lookup")
        if key in self.lookup._v_children:
            return self.lookup._v_children[key]
        raise AttributeError(f"key {key} not found")

    def import_omx(self, otherfile, tablenames, rowslicer=None, colslicer=None):
        oth = OMX(otherfile, mode="r")
        if tablenames == "*":
            tablenames = oth.data._v_children.keys()
        for tab in tablenames:
            if rowslicer is None and colslicer is None:
                self.add_matrix(tab, oth.data._v_children[tab][:])
            else:
                self.add_matrix(tab, oth.data._v_children[tab][rowslicer, colslicer])


def _shrink_bitwidth(v, dtype_shrink=32):
    if dtype_shrink <= 32:
        if v.dtype == "float64":
            v = v.astype("float32")
        elif (
            v.dtype == "int64" and v.max() < 2_147_483_648 and v.min() > -2_147_483_648
        ):
            v = v.astype("int32")
    return v


def convert_omx(
    existing_filename: str,
    new_filename: str,
    complevel=3,
    complib="blosc2:zstd",
    dtype_shrink=32,
    part=0,
    n_part=1,
) -> float:
    """
    Convert an existing OMX file using different data types and filters.

    This function will read an existing OMX file and write it back out
    in a new format.  This is useful for updating old OMX files to
    the new format, which may be more efficient.

    Parameters
    ----------
    existing_filename : str
        The filename of the existing OMX file to convert.
    new_filename : str
        The filename of the new OMX file to write.
    complevel : int, default 3
        Compression level.
    complib : str, default 'blosc2:zstd'
        Compression library.
    dtype_shrink : int, default 32
        The maximum bitwidth to use for integer and float types.
    part : int, default 0
        The part of the file to start with.
    n_part : int, default 1
        The number of partitions to create.

    Returns
    -------
    float
        The time in seconds it took to convert the file.
    """
    print(f"[green]converting[/green] {existing_filename} to {new_filename}...")
    start = time.time()
    with OMX(existing_filename, mode="r") as old_omx:
        if part != 0 or n_part != 1:
            newbase, nextext = os.path.splitext(new_filename)
            new_filename = newbase + f".part{part:03d}" + nextext
        if new_filename == existing_filename and new_filename.endswith(".omx"):
            new_filename = new_filename[:-4] + ".omxz"
        with OMX(new_filename, mode="w") as new_omx:
            keys = sorted(old_omx.data._v_children)
            for name in keys[part::n_part]:
                v = old_omx.data._v_children[name][:]
                v = _shrink_bitwidth(v, dtype_shrink)
                new_omx.add_matrix(
                    name,
                    v,
                    complevel=complevel,
                    complib=complib,
                )
            keys = sorted(old_omx.lookup._v_children)
            for name in keys[part::n_part]:
                v = old_omx.lookup._v_children[name][:]
                v = _shrink_bitwidth(v, dtype_shrink)
                new_omx.add_lookup(
                    name,
                    v,
                    complevel=complevel,
                    complib=complib,
                )
    return time.time() - start


@app.command(name="omx")
def convert_multiple_omx(
    glob_pattern: Annotated[
        str,
        typer.Argument(help="file name or glob pattern for the OMX files to convert"),
    ] = "*.omx",
    complevel: Annotated[int, typer.Argument(help="compression level")] = 3,
    complib: Annotated[str, typer.Argument(help="compression library")] = "blosc2:zstd",
    dtype_shrink: Annotated[
        int, typer.Option(help="maximum bitwidth to use for integer and float types")
    ] = 32,
    n_processes: Annotated[
        int, typer.Option(help="number of processes to use while converting files")
    ] = 8,
    out_dir: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "directory name for the resulting reprocessed file(s), if not "
                "provided any '.omx' suffix is stripped from the original file "
                "name(s) and a '.omxz' suffix is added"
            ),
        ),
    ] = None,
) -> None:
    """
    Convert one or more OMX files using different data types and filters.
    """
    start = time.time()
    import glob
    from multiprocessing import Pool

    c = console.Console()

    # start worker processes
    with Pool(processes=n_processes) as pool:
        multiple_results = []
        results_outstanding = 0

        for f in glob.glob(str(glob_pattern)):
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                new_filename = os.path.join(out_dir, os.path.basename(f))
            else:
                new_filename = f + "z"
            r = pool.apply_async(
                convert_omx,
                (f, new_filename, complevel, complib, dtype_shrink),
            )
            multiple_results.append((r, f, new_filename))
            results_outstanding += 1

        while results_outstanding > 0:
            c.status(f"converting {results_outstanding} files...")
            time.sleep(1)
            for i, (res, f, new_filename) in enumerate(multiple_results):
                if res.ready():
                    t = res.get(timeout=60)
                    print(
                        f"[green]converted[/green] {f} to {new_filename} in"
                        f" {t:.2f} seconds."
                    )
                    results_outstanding -= 1
                    del multiple_results[i]
                    break

    wall_time = time.time() - start
    c.status(f"converted all files in {wall_time:.2f} seconds.")
