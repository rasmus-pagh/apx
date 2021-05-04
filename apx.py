import urllib.request
import os.path

class DataFile:

  url_prefix = 'https://raw.githubusercontent.com/rasmus-pagh/apx/main/data/'
  data_directory = 'data/'
  graph_files = ['routes.txt','petersen.txt','petersenstar.txt','star.txt','clique.txt','cycles.txt','lotr.txt','karate.txt', 'noisybiclique.txt']

  def __init__(self, filename):
    if not os.path.exists(self.data_directory):
        os.makedirs(self.data_directory)
    if not os.path.isfile(self.data_directory + filename):
      urllib.request.urlretrieve(self.url_prefix + filename, self.data_directory + filename)  
    if not os.path.isfile(self.data_directory + filename):
      raise ValueError('Unknown file: {filename}\nKnown files: {files}')
    else: 
      self.f = open(self.data_directory + filename, "r")

  def __iter__(self):
    return self

  def __next__(self):
    line = self.f.readline()
    if line == '': 
      raise StopIteration
    return [ x for x in line.rstrip('\n').split(' ') if x != '']
    

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import re

class LinearProgram:

  term_re = re.compile('([-+]?)(\d*\.\d+|\d+)?\*?(\w+)')

  def __init__(self, objective_type = 'max'):
    self.objective_type = objective_type
    self.row_numbers = []
    self.column_numbers = []
    self.entry_weights = []
    self.bounds = []
    self.objective = []
    self.map_name_column = {}
    self.map_column_name = {}
    self.map_name_row = {}
    self.map_row_name = {}
    self.num_columns = 0
    self.num_rows = 0

  def column_number(self, column_name):
    if not column_name in self.map_name_column:
      self.map_name_column[column_name] = self.num_columns
      self.map_column_name[self.num_columns] = column_name
      self.num_columns += 1
    return self.map_name_column[column_name]

  def parse_expression(self, x):
    map_name_weight = {}
    for match in self.term_re.finditer(x.replace(" ", "")):
      if match.group(1) == '-':
        sign = -1
      else:
        sign = 1
      if match.group(2) is None:
        weight = 1.0
      else:
        weight = float(match.group(2))
      map_name_weight[match.group(3)] = sign * weight
    return map_name_weight

  def add_constraint(self, sparse_row, b, name = None):
    if isinstance(sparse_row, str):
      map_name_weight = self.parse_expression(sparse_row)
    else:
      map_name_weight = sparse_row
    self.bounds.append(b)
    for column_name in map_name_weight:
      self.row_numbers.append(self.num_rows)
      self.column_numbers.append(self.column_number(column_name))
      self.entry_weights.append(map_name_weight[column_name])
    if name is None:
      i = self.num_rows + 1
      while 'y'+str(i) in self.map_name_row: # Find unique row name
        i += 1
      name = 'y'+str(i)
    assert(name not in self.map_name_row)
    self.map_name_row[name] = self.num_rows
    self.map_row_name[self.num_rows] = name
    self.num_rows += 1

  def set_objective(self, sparse_objective):
    if isinstance(sparse_objective, str):
      sparse_objective = self.parse_expression(sparse_objective)
    for column_name in sparse_objective: # Ensure that all names map to columns
      self.column_number(column_name)
    self.objective = [sparse_objective.get(self.map_column_name[j], 0.0) for j in range(self.num_columns)]

  def to_string(self):
    A = coo_matrix((self.entry_weights, (self.row_numbers, self.column_numbers))).todense()
    if self.objective_type == 'max':
      return f'Maximize c x under A x <= b, x >= 0, where\nA={A}\nb={self.bounds}\nc={self.objective}'
    else:
      return f'Minimize b y under A y >= c, y >= 0, where\nA={A}\nb={self.objective}\nc={self.bounds}'
  
  def dual(self):
    if self.objective_type == 'max':
      res = LinearProgram('min')
    else:
      res = LinearProgram('max')    
    res.entry_weights = self.entry_weights.copy()
    res.row_numbers = self.column_numbers.copy()
    res.column_numbers = self.row_numbers.copy()
    res.bounds = self.objective.copy()
    res.objective = self.bounds.copy()
    res.map_name_column = self.map_name_row.copy()
    res.map_column_name = self.map_row_name.copy()
    res.map_name_row = self.map_name_column.copy()
    res.map_row_name = self.map_column_name.copy()
    res.num_columns = self.num_rows
    res.num_rows = self.num_columns
    return res

  def solve(self):
    A = coo_matrix((self.entry_weights, (self.row_numbers, self.column_numbers))).todense()
    b = np.array(self.bounds)
    c = np.array(self.objective)
    if self.objective_type == 'max':
      sign = -1
    elif self.objective_type == 'min':
      sign = 1
    else:
      raise ValueError(f'Unknown objective type: {self.objective}')
    res = linprog(sign * c, A_ub=-sign*A, b_ub=-sign*b, options={'sym_pos': False, 'lstsq': True})
    solution_dict = {}
    for column_name in self.map_name_column:
      solution_dict[column_name] = sign * res.x[self.map_name_column[column_name]]
    return sign * res.fun, solution_dict
