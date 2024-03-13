import os
import yaml
import openpyxl

import matplotlib.pyplot as plt


plt.ioff()
plt.switch_backend('agg')


class XlsBook():
    def __init__(self, labels, sheet_name='log'):
        self.labels = labels
        self.book = openpyxl.Workbook()
        self.sheet = self.book.create_sheet(sheet_name, 0)
        self.sheet.append(labels)

    def write(self, values):
        if len(values) != len(self.labels):
            raise ValueError('Inputs of logger does not match the length of the labels.')
        self.sheet.append(values)

    def save(self, save_path):
        self.book.save(save_path)


class QueueList():
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.list = []

    def add(self, x):
        if len(self.list) == self.max_size:
            self.list.pop(0)
        self.list.append(x)

    def remove(self, i):
        self.list.pop(i)