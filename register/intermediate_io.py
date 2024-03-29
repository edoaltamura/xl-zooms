# -*- coding: utf-8 -*-
"""Tools for handling intermediate analysis products

This package contains classes that use the `pickle` library to dump
and read back into memory intermediate products of the analysis.

Generally, quantities which take a long time to compute and do not
need to be freshly calculated every time should be dumped to disk and
read back when requires.

IMPORTANT - (abstract) Class attributes are lost in pickling
When you pickled the instance you haven't pickled the class attributes,
just the instance attributes. So when you unpickle it you get just the
instance attributes back.
"""
import os
from warnings import warn
from pandas import DataFrame, read_pickle

try:
    import _pickle as pickle

    # `HIGHEST_PROTOCOL` attribute not defined in `_pickle`
    pickle.HIGHEST_PROTOCOL = -1

except ModuleNotFoundError:
    import pickle

from .auto_parser import xlargs
from .static_parameters import default_output_directory

# Make sure the `intermediate` directory exists in the output directory
if not os.path.isdir(os.path.join(default_output_directory, 'intermediate')):
    os.makedirs(os.path.join(default_output_directory, 'intermediate'))


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class CustomPickler:
    def __init__(self, filename: str, relative_path: bool = False):

        if relative_path:
            self.filename = os.path.join(
                default_output_directory,
                'intermediate',
                filename
            )
        else:
            self.filename = filename

    def large_file_warning(self) -> None:
        file_size_b = os.path.getsize(self.filename)
        if not xlargs.quiet and file_size_b > 524288000:
            warn(
                (
                    '[io] Detected file larger than 500 MB! '
                    'Trying to import all contents of pkl to memory at once. '
                    'If the file is large, you may run out of memory or degrade the '
                    'performance. You can use the `MultiObjPickler.get_pickle_generator` '
                    'to access a generator, which returns only one pickled object at a '
                    'time.'
                ),
                category=ResourceWarning
            )


class SingleObjPickler(CustomPickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj):
        with open(self.filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        if not xlargs.quiet:
            file_size = sizeof_fmt(
                os.path.getsize(
                    self.filename
                )
            )
            print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def load_from_pickle(self):
        self.large_file_warning()

        return pickle.load(self.filename)


class MultiObjPickler(CustomPickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj_collection):
        with open(self.filename, 'wb') as output:  # Overwrites any existing file.
            for obj in obj_collection:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        if not xlargs.quiet:
            file_size = sizeof_fmt(
                os.path.getsize(
                    self.filename
                )
            )
            print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def get_pickle_generator(self):
        """ Unpickle a file of pickled data. """
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"File {self.filename} not found.")

        if not xlargs.quiet:
            file_size = sizeof_fmt(
                os.path.getsize(
                    self.filename
                )
            )
            print(f"[io] Loading from pkl [{file_size:s}]: {self.filename:s}...")

        with open(self.filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def load_from_pickle(self):
        self.large_file_warning()
        collection_pkl = []
        for obj in self.get_pickle_generator():
            collection_pkl.append(obj)
        return collection_pkl


class DataframePickler(CustomPickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dump_to_pickle(self, obj: DataFrame):
        # Save data in pickle format (saves python object)
        obj.to_pickle(self.filename)

        # Save data in text format (useful for consulting)
        obj.to_csv(self.filename.replace('pkl', 'txt'), header=True, index=False, sep=',', mode='w')

        if not xlargs.quiet:
            file_size = sizeof_fmt(
                os.path.getsize(
                    self.filename
                )
            )
            print(f"[io] Object saved to pkl [{file_size:s}]: {self.filename:s}")

    def load_from_pickle(self):
        self.large_file_warning()

        return read_pickle(self.filename)
