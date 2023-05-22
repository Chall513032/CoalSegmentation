import os
import numpy as np
from PIL import Image


'''
Figure information
'''
pwd      = os.getcwd()
path     = os.path.join(pwd, 'results')

scale3   = [1, 1, 1]
resamples= 50
resample = [resamples//scale3[0], resamples//scale3[1], resamples//scale3[2]]
files    = os.listdir(path)[::resample[2]]
scale    = 1
numslice = len(files)
size     = np.asarray(Image.open(os.path.join(path, files[0])),np.uint8)[::resample[0], ::resample[1]].shape
size     = [size[1], size[0]]
meshnum  = [int(size[0]//scale), int(size[1]//scale), int(numslice), 10]
warpnum  = 1
weftnum  = 1

'''
Mesh information
'''
grid_ele  = np.zeros(meshnum)                                        # 0 for elenum, 1 for part ID, 2-9 for node numbers
grid_node = np.zeros([meshnum[0]+1, meshnum[1]+1, meshnum[2]+1, 4])  # 0 for number, 1-3 for coords
iele      = 1
pixel_l   = 4.9819e-3  # 4.98e-3
pixel     = [pixel_l*(resamples * (resamples/resample[0])),
             pixel_l*(resamples * (resamples/resample[1])),
             pixel_l*(resamples * (resamples/resample[1]))]


for islice, ifile in enumerate(files):
    print('Image:'+str(islice+1))

    '''
    Start reading image
    '''
    image1= Image.open(os.path.join(path, ifile))
    grey1 = np.array(image1.convert('L'), 'f').transpose([1, 0])
    grey1 = grey1[::resample[0], ::resample[1]]

    for i in range(meshnum[0]):
        for j in range(meshnum[1]):
            '''
            Compress image
            '''

            sumv1 = grey1[i,j]

            if sumv1 != 0:
                grid_ele[i, j, islice, 1] = sumv1
            # else:
            #     grid_ele[i, j, islice, 1] = sumv1+102

            '''
            Mesh grid
            '''
            grid_node[i  ,j  ,islice,  0] = islice   *(meshnum[0]+1)*(meshnum[1]+1) +  j   *(meshnum[0]+1) + i + 1
            grid_node[i+1,j  ,islice,  0] = islice   *(meshnum[0]+1)*(meshnum[1]+1) +  j   *(meshnum[0]+1) + i + 2
            grid_node[i+1,j+1,islice,  0] = islice   *(meshnum[0]+1)*(meshnum[1]+1) + (j+1)*(meshnum[0]+1) + i + 2
            grid_node[i  ,j+1,islice,  0] = islice   *(meshnum[0]+1)*(meshnum[1]+1) + (j+1)*(meshnum[0]+1) + i + 1
            grid_node[i  ,j  ,islice+1,0] =(islice+1)*(meshnum[0]+1)*(meshnum[1]+1) +  j   *(meshnum[0]+1) + i + 1
            grid_node[i+1,j  ,islice+1,0] =(islice+1)*(meshnum[0]+1)*(meshnum[1]+1) +  j   *(meshnum[0]+1) + i + 2
            grid_node[i+1,j+1,islice+1,0] =(islice+1)*(meshnum[0]+1)*(meshnum[1]+1) + (j+1)*(meshnum[0]+1) + i + 2
            grid_node[i  ,j+1,islice+1,0] =(islice+1)*(meshnum[0]+1)*(meshnum[1]+1) + (j+1)*(meshnum[0]+1) + i + 1

            grid_ele[i,j,islice,0] = iele
            grid_ele[i,j,islice,2] = grid_node[i  ,j  ,islice,  0]
            grid_ele[i,j,islice,3] = grid_node[i+1,j  ,islice,  0]
            grid_ele[i,j,islice,4] = grid_node[i+1,j+1,islice,  0]
            grid_ele[i,j,islice,5] = grid_node[i  ,j+1,islice,  0]
            grid_ele[i,j,islice,6] = grid_node[i  ,j  ,islice+1,0]
            grid_ele[i,j,islice,7] = grid_node[i+1,j  ,islice+1,0]
            grid_ele[i,j,islice,8] = grid_node[i+1,j+1,islice+1,0]
            grid_ele[i,j,islice,9] = grid_node[i  ,j+1,islice+1,0]

            iele += 1

    for i in range(meshnum[0]+1):
        for j in range(meshnum[1]+1):
            grid_node[i,j,islice  ,1], grid_node[i,j,islice  ,2], grid_node[i,j,islice  ,3] = i*pixel[0], \
                                                                                              j*pixel[1], \
                                                                                              islice*pixel[2]
            grid_node[i,j,islice+1,1], grid_node[i,j,islice+1,2], grid_node[i,j,islice+1,3] = i*pixel[0], \
                                                                                              j*pixel[1], \
                                                                                              (islice+1)*pixel[2]

# grid_ele[0, :, :, 1] = 0
townum = len(np.unique(grid_ele[:, :, :, 1]))

NodeSave = [[] for _ in range(townum + 1)]  
for idx in range(1, townum + 1):
    NodeTemp = []
    for k in range(meshnum[2]):
        for i in range(meshnum[0]):
            for j in range(meshnum[1]):
                if grid_ele[i, j, k, 1] == idx:
                    temp = []
                    for m in range(2, 10):
                        temp.append(grid_ele[i, j, k, m])
                    NodeTemp.append(temp)
    NodeTemp = np.unique(np.array(NodeTemp, "int"))
    NodeSave[idx] = NodeTemp


ElSave = [[] for _ in range(townum + 1)] 
for idx in range(1, townum + 1):
    ElTemp = []
    for k in range(meshnum[2]):
        for i in range(meshnum[0]):
            for j in range(meshnum[1]):
                if grid_ele[i, j, k, 1] == idx:
                    ElTemp.append(grid_ele[i, j, k, 0])
    ElTemp = np.array(ElTemp, "int")
    ElSave[idx] = ElTemp


'''
Write to inp file
'''
print('Start writing inp file')
filename = 'NDEL.inp'
with open(filename, 'w') as f:
    f.write("*Heading\n")
    f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
    f.write("*Part, name=Part-1\n")
    f.write("*Node\n")
    for k in range(meshnum[2] + 1):
        for i in range(meshnum[0] + 1):
            for j in range(meshnum[1] + 1):
                # f.write(str('%8d' % grid_node[i, j, k, 0]) + ',')
                f.write(str('%10f' % grid_node[i, j, k, 1]) + ',')
                f.write(str('%10f' % grid_node[i, j, k, 2]) + ',')
                f.write(str('%10f' % grid_node[i, j, k, 3]) + '\n')


    f.write("*Element, type=C3D8R\n")
    for k in range(meshnum[2]):
        for i in range(meshnum[0]):
            for j in range(meshnum[1]):
                if grid_ele[i, j, k, 1] > 0:
                    f.write(str('%7d' % grid_ele[i, j, k, 0]))
                    for m in range(2, 10):
                        f.write(',' + str('%7d' % grid_ele[i, j, k, m]))
                    f.write('\n')

    for idx in range(1, townum + 1):
        f.write(f"*Nset, nset=Set-{idx}\n")
        pos = 0
        for i in range(len(NodeSave[idx]) - 1):
            if pos < 15:
                f.write(str('%7d' % NodeSave[idx][i]) + ',')
                pos += 1
            else:
                f.write(str('%7d' % NodeSave[idx][i] + '\n'))
                pos = 0
        # f.write("\n")
        if len(NodeSave[idx]) == 0:
            f.write("\n")
        else:
            f.write(str('%7d' % NodeSave[idx][i + 1] + '\n'))

        f.write(f"*Elset, elset=Set-{idx}\n")
        pos = 0
        for i in range(len(ElSave[idx]) - 1):
            if pos < 15:
                f.write(str('%7d' % ElSave[idx][i]) + ',')
                pos += 1
            else:
                f.write(str('%7d' % ElSave[idx][i] + '\n'))
                pos = 0
        # f.write("\n")
        if len(ElSave[idx]) == 0:
            f.write("\n")
        else:
            f.write(str('%7d' % ElSave[idx][i + 1] + '\n'))

    for idx in range(0, townum + 1):
        f.write(f"*Solid Section, elset=Set-{idx}, material=Material-0\n")
        f.write(",\n")

    f.write("*End Part\n")
    f.write("*Assembly, name=Assembly\n")
    f.write("*Instance, name=Part-1-1, part=Part-1\n")
    f.write("*End Instance\n")
    f.write("*End Assembly\n")
    # for idx in range(0, townum + 1):
    #     f.write(f"*Material, name=Material-{idx}\n")
    f.write(f"*Material, name=Material-0")
