import os
from warnings import warn

try:
    import _pickle as pickle
except ModuleNotFoundError:
    import pickle

from .auto_parser import args
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


class MultiObjPickler:
    def __init__(self, filename: str, relative_path: bool = False):

        if relative_path:
            self.filename = os.path.join(
                default_output_directory,
                'intermediate',
                filename
            )
        else:
            self.filename = filename

    def dump_to_pickle(self, obj):
        with open(self.filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        if not args.quiet:
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

        if not args.quiet:
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

        file_size_b = os.path.getsize(self.filename)
        if not args.quiet and file_size_b > 52428800:
            warn(
                (
                    '[io] Detected file larger than 50 MB! '
                    'Trying to import all contents of pkl to memory at once. '
                    'If the file is large, you may run out of memory or degrade the '
                    'performance. You can use the `MultiObjPickler.get_pickle_generator` '
                    'to access a generator, which returns only one pickled object at a '
                    'time.'
                ),
                category=ResourceWarning
            )

        retrieve_pickled = []
        for obj in self.get_pickle_generator():
            retrieve_pickled.append(obj)

        return retrieve_pickled
