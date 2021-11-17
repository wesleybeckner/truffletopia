import random
from scipy.stats import gamma, norm
from os.path import dirname, join
import pandas as pd
import numpy as np

class Truffle():
  """
  Truffle products made by truffletopia
  """
  def __init__(self, base_cake=None, truffle_type=None, primary_flavor=None,
               secondary_flavor=None, color_group=None):
    self.base_cake = base_cake
    self.truffle_type = truffle_type
    self.primary_flavor = primary_flavor
    self.secondary_flavor = secondary_flavor
    self.color_group = color_group

  def set_attributes(self):
    """
    Sets the product attributes by randomly selecting from the datafile
    operations.csv
    """
    operations = load_data("operations.csv")
    for uni in operations.Category.unique():
      att = random.choice(list(operations.loc[operations.Category == uni]
                               .Classification))
      self.__dict__[uni] = att

  def get_params(self):
    """
    Print out the parameters of the truffle object
    """
    for param in self.__dict__:
      print(f"{param}: {self.__dict__[param]}")

class UnitOperation():
  """
  Unit Operations in truffletopia
  """
  def __init__(self, name=None, rate=None, std=None, distribution=None):
    self.name = name
    self.rate = rate
    self.std = std
    self.distribution = distribution

  def set_attributes(self, truffle, line_effect=None):
    """
    Sets the UnitOps attributes according to the characteristics of the product
    being made. line_effect is a 2 element list that contains the multiplicative
    effect of the line on mu and sigma, applied at the final calculation of mu, sigma
    """
    mus = []
    sigs = []
    dists = []
    name = ''
    samp_size = 12
    class_weights = np.array([80, 8, 4, 4, 4])
    operations = load_data("operations.csv")

    for cat, uni in truffle.__dict__.items():
      name += uni[:3]
      mus.append(operations.loc[operations['Classification'] == uni]
                 ['Mean'].values[0])
      sigs.append(operations.loc[operations['Classification'] == uni]
                  ['Std'].values[0])
      dists.append(operations.loc[operations['Classification'] == uni]
                   ['Distribution'].values[0])
    mus = np.array(mus)
    if line_effect:
        mu = round(line_effect[0]*(class_weights.dot(mus)/np.sum(class_weights)), 4)
        sig = round(line_effect[1]*class_weights.dot(sigs)/np.sum(class_weights)*.1, 4)
    else: 
        mu = round(class_weights.dot(mus)/np.sum(class_weights), 4)
        sig = round(class_weights.dot(sigs)/np.sum(class_weights)*.1, 4)

    if 'weibull' not in dists :
      dist = norm
    else:
      dist = gamma

    self.rate = mu
    self.std = sig
    self.distribution = dist
    if not line_effect:
      self.name = name

  def run(self, quantity=1):
    """
    run the UnitOp

    Parameters
    ----------
    quantity: int
      the number of samples to draw from the distribution, default 1

    Returns
    -------
    rates: list of floats
      rate of the UnitOperation drawn from the distribution set by the UnitOp

    """
    if self.distribution.name == 'gamma':
      rates = self.distribution.rvs(size=quantity, loc=self.rate,
                                    scale=self.std, a=1)
    else:
      rates = self.distribution.rvs(size=quantity, loc=self.rate,
                                    scale=self.std)

    return rates

  def get_params(self):
    """
    Print out the parameters of the truffle object
    """
    for param in self.__dict__:
      print(f"{param}: {self.__dict__[param]}")

def load_data(data_file_name):
    """Loads data from module_path/data/data_file_name.
    Parameters
    ----------
    data_file_name : String
        Name of csv or dill file to be loaded from
        module_path/data/data_file_name. For example 'operations.csv'.

    Returns
    -------
    data : DataFrame
        A data frame.
    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data', data_file_name), 'rb') as csv_file:
        data = pd.read_csv(csv_file, encoding='latin1')
    return data
