import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2


'''m = [
[ 1,    0.88,  0.84, 0.86 ],
[ 0.88,    1,  0.45, 0.07 ],
[ 0.84, 0.45,  1,    0.43 ],
[ 0.86, 0.07,  0.43,    1 ],
]

font = { 'family' : 'sans serif', 'size': 16 }
#plt.rc('font', **font)
plt.matshow(m, cmap=plt.cm.Blues, vmin=0, vmax=1)

groups = [ "LiRe", "LoRe", "BeRe", "LDA" ]
 
x_pos = np.arange(len(groups))
plt.xticks(x_pos,groups)
 
y_pos = np.arange(len(groups))
plt.yticks(y_pos,groups)
colorbar()
plt.title( "Tools similarity matrix" )
plt.show()'''

# Plot venn for Jaccard Index
 
# First way to call the 2 group Venn diagram:
'''plt.subplot(211)
venn2(subsets = (3, 4, 2), set_labels = ('Tool1: file types', 'Tool2: file types'))
plt.title( "Jaccard Index" )
font = { 'family' : 'sans serif', 'size': 16 }
plt.rc('font', **font)
plt.show()'''
 
# Second way
#venn2([set(['A', 'B', 'C', 'D']), set(['D', 'E', 'F'])])
#plt.show()


plt.subplot(211)
venn2(subsets = (4, 4, 3), set_labels = ('Top-k predicted classes', 'Actual k classes'))
plt.title( "Top-k accuracy" )
font = { 'family' : 'sans serif', 'size': 16 }
plt.rc('font', **font)
plt.show()
