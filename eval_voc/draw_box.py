# -*- coding:utf-8 -*-
#!/usr/bin/env Python
import os
import shutil
from PIL import Image, ImageDraw
import pdb
#==========================建立索引文件夹==========================



    
dict1 = {'1':'aero','2':'bike','3':'bird','4':'boat','5':'bottle','6':'bus','7':'car','8':'cat','9':'chair','10':'cow','11':'table','12':'dog','13':'horse','14':'mbike','15':'person','16':'plant','17':'sheep','18':'sofa','19':'train','20':'tv'}

# for i in range(1, 21):
#         os.mkdir(Path1_A+str(i))

def draw(d, A, score='GT:',cls=' ',color='red'):
    tex = score+cls
    line = 2
    x, y = A[0], A[1]
    x_1, y_1 = A[2], A[3]

    for i in range(1, line + 1):
        d.rectangle((x + (line - i), y + (line - i), x_1 + i, y_1 + i), outline=color)
    if not score == 'GT:':
        d.text([x,y],tex,color)
    else:
        d.text([x_1,y_1],tex,color)

#==========================main==========================
if __name__ == "__main__":
    result = []
    import sys
    #alllist=os.listdir("/home/tucoder/Desktop/demo1_result")
    lines = open(sys.argv[1],'r').readlines()
    root = '/cephfs/person/jawnrwen/to_WenWei/ssd.pytorch/data/VOC/VOCdevkit/VOC2007/JPEGImages/'
    #pdb.set_trace()
    save_path = 'save'
    if not os.path.exists(save_path): os.mkdir(save_path)
    filename = ' '
    cur_img = lines[0].strip().split(':')[-1]
    for line in lines:
        if not line.strip(): continue
        line = line.strip()
        if 'img:' in line:
            if cur_img != line.split(':')[-1]:
                cur_img = line.split(':')[-1]
                im.save(os.path.join(save_path,filename.split('/')[-1]),quality=95)

            filename = root + line.split(':')[-1]+'.jpg'
            im = Image.open(filename)
            d = ImageDraw.Draw(im)
        elif 'gted:' in line:
            cur_A = [float(i) for i in line.split()[1:-1]]
            c = dict1[str(int(line.split()[-1])+1)]
            draw(d,cur_A,cls=c,color='blue')
        elif 'pred:' in line:
            cur_A = [float(i) for i in line.split()[3:]]
            score = line.split()[2].split('(')[-1].split(')')[0]
            c = line.split()[1]
            draw(d,cur_A,score=score,cls=c)
        



    '''
        set1 = set()
        string = '' # 每幅图片包含其class属性
        str1 = filename[:filename.rindex(".")]
        with open(filename, 'r') as f:
            for line in f.readlines():
                list1 = line.split()
                if float(list1[1]) >= 0.80: # 类别置信度>=0.8，判定其为真
                    A.add(list1[0])
                    Path_1 = str1+".jpg"
                    Path_2 = Path1_A+'/'+list1[0]+"/"+str1+".jpg"
                    if list1[0] not in set1:
                        set1.add(list1[0]) 
                        draw(Path_1, float(list1[2]), float(list1[3]), float(list1[4]), float(list1[5]), Path_2)
                        string = string +list1[0]+"-"
                    else:
                        draw(Path_2, float(list1[2]), float(list1[3]), float(list1[4]), float(list1[5]), Path_2)
        set1.clear()
        List = []
    for str_i in list(A):
        List.append(dict1[str_i])
    print(List)
    with open(Path2_A+'text', 'w') as f:
        f.write(str(List))
    '''
