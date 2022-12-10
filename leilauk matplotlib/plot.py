import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/Users/asadtariq/Downloads/Python_work/Python_work/leilauk matplotlib/braindatanew.csv')
df.head()

f1= df[["1","17"]]
f2= df[["2","18"]]
f3 = df[["3","19"]]
f4 = df[["4","20"]]
f5 = df[["5","21"]]
f6 = df[["6","22"]]
f7 = df[["7","23"]]
f8 = df[["8","24"]]
f9 = df[["9","25"]]
f10 = df[["10","26"]]
f11 = df[["11","27"]]
f12 = df[["12","28"]]
f13 = df[["13","29"]]
f14 = df[["14","30"]]
f15 = df[["15","31"]]
f16 = df[["16","32"]]
clr=['blue','red']

fig, axes = plt.subplots(nrows=4, ncols=4)
fig.set_figheight(8)
fig.set_figwidth(12)
fig.supxlabel('Time [ms]')
fig.supylabel('Resting Membrane Potential [mV]')

plt.setp(axes, xticks=[0,1000], xticklabels=[0,1000],
        yticks=[-75,-50,-25])
f1.plot(ax=axes[0,0],legend=None,color=clr).set_title("f: 1")
f2.plot(ax=axes[0,1],legend=None,color=clr).set_title("f: 2")
f3.plot(ax=axes[0,2],legend=None,color=clr).set_title("f: 3")
f4.plot(ax=axes[0,3],legend=None,color=clr).set_title("f: 4")

f5.plot(ax=axes[1,0],legend=None,color=clr).set_title("f: 5")
f6.plot(ax=axes[1,1],legend=None,color=clr).set_title("f: 6")
f7.plot(ax=axes[1,2],legend=None,color=clr).set_title("f: 7")
f8.plot(ax=axes[1,3],legend=None,color=clr).set_title("f: 8")

f9.plot(ax=axes[2,0],legend=None,color=clr).set_title("f: 9")
f10.plot(ax=axes[2,1],legend=None,color=clr).set_title("f: 10")
f11.plot(ax=axes[2,2],legend=None,color=clr).set_title("f: 11")
f12.plot(ax=axes[2,3],legend=None,color=clr).set_title("f: 12")

f13.plot(ax=axes[3,0],legend=None,color=clr).set_title("f: 13")
f14.plot(ax=axes[3,1],legend=None,color=clr).set_title("f: 14")
f15.plot(ax=axes[3,2],legend=None,color=clr).set_title("f: 15")
f16.plot(ax=axes[3,3],legend=None,color=clr).set_title("f: 16")
plt.show()