import collections
import copy
import json
import math
import re
import six


class Config(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               root='./Data/',
               labeled_path='./Labeled/',
               unlabeled_path='./Unlabeled/',
               test_path='./Test/',
               validation_path='./Validation/',
               map_path='./addr_to_index.json',
               model_path='./model.t7',
               hosts=['192.168.137.183', '192.168.137.19'],
               port=8000,
               n_address=32,
               x_min=0,
               x_max=1,
               y_min=0,
               y_max=1):
    """Constructs BertConfig.
    Args:
      root: Data root
    """
    self.root = root
    self.labeled_path = labeled_path
    self.unlabeled_path = unlabeled_path
    self.test_path = test_path
    self.validation_path = validation_path
    self.map_path = map_path
    self.model_path = model_path
    self.hosts = hosts
    self.port = port
    self.n_address = n_address
    self.x_min=x_min
    self.x_max=x_max
    self.y_min=y_min
    self.y_max=y_max

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `Config` from a Python dictionary of parameters."""
    config = Config()
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `Config` from a json file of parameters."""
    return cls.from_dict(json.load(open(json_file)))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2) + "\n"
    
  def to_file(self):
    """Serializes this instance to a JSON string."""
    with open('config.json', 'w') as f:
      f.write(self.to_json_string())
    
if __name__ == '__main__':
    config = Config.from_json_file('config.json')
    config.to_file()
