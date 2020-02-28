import numpy as np
import datetime

# Taken from Stackoverflow

def DecYear(y, m, d):
     date = datetime.date(y, m, d)
     start = datetime.date(date.year, 1, 1).toordinal()
     year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
     return date.year + float(date.toordinal() - start) / year_length
