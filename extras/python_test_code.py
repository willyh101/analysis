class DistinctError(ValueError):
       """Raised when duplicate value is added to a distinctdict."""
       
class distinctdict(dict):
    """Dictionary that does not accept duplicate values."""
    def __setitem__(self, key, value):
        if value in self.values():
            if (
                (key in self and self[key] != value) or
                key not in self
            ):
                raise DistinctError(
                    "This value already exists for different key"
                )
        super().__setitem__(key, value)
        
class Folder(list):
       def __init__(self, name):
           self.name = name
       def dir(self, nesting=0):
           offset = "  " * nesting
           print('%s%s/' % (offset, self.name))
           for element in self:
               if hasattr(element, 'dir'):
                   element.dir(nesting + 1)
               else:
                   print("%s  %s" % (offset, element))