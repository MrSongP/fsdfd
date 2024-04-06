import os

import numpy as np



celeba = '/root/CelebA/reality'
pggan = '/root/pggan'
stargan = '/root/stargan'
stylegan = '/root/stylegan'
ffhq = '/root/ffhq'
ddpm = '/root/DDPM'
stylegan2 = '/root/stylegan2'
ldm = '/root/LDM'

savedir = './'

file_list = []
label_list = []


for i ,img in enumerate(os.listdir(celeba)[:3800]):
    file_list = file_list +  [celeba+'/'+img]
    label_list = label_list + [0]

for i ,img in enumerate(os.listdir(pggan)[:3800]):
    file_list = file_list + [pggan + '/' + img]
    label_list = label_list + [1]

for i ,img in enumerate(os.listdir(stargan)[:3800]):
    file_list = file_list + [stargan + '/' + img]
    label_list = label_list + [2]

for i ,img in enumerate(os.listdir(stylegan)[:3800]):
    file_list = file_list + [stylegan + '/' + img]
    label_list = label_list + [3]

# for i ,img in enumerate(os.listdir(ddpm)[:3800]):
#     file_list = file_list + [ddpm + '/' + img]
#     label_list = label_list + [4]



fo = open(savedir + 'base' + ".json", "w")

fo.write('{"image_names": [')
fo.writelines(['"%s",' % item for item in file_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item for item in label_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("base -OK" )

file_list = []
label_list = []

# for i ,img in enumerate(os.listdir(celeba)[-2000:]):
#     file_list = file_list + [celeba + '/' + img]
#     label_list = label_list + [5]

for i ,img in enumerate(os.listdir(ddpm)[:2000]):
    file_list = file_list + [ddpm + '/' + img]
    label_list = label_list + [4]

for i ,img in enumerate(os.listdir(stylegan2)[:2000]):
    file_list = file_list + [stylegan2 + '/' + img]
    label_list = label_list + [5]

for i ,img in enumerate(os.listdir(ldm)[:2000]):
    file_list = file_list + [ldm + '/' + img]
    label_list = label_list + [6]

# for i ,img in enumerate(os.listdir(ffhq)[:20]):
#     file_list = file_list + [ffhq + '/' + img]
#     label_list = label_list + [7]

fo = open(savedir + 'novel' + ".json", "w")

fo.write('{"image_names": [')
fo.writelines(['"%s",' % item for item in file_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item for item in label_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell() - 1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("novel -OK" )


