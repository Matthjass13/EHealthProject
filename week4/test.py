import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])


dates = pd.date_range("20130101", periods=6)

#Out
#Out DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
#Out               '2013-01-05', '2013-01-06'],
  #Out            dtype='datetime64[ns]', freq='D')


df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))

df