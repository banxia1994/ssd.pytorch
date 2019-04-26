import json
import sys

ori_ = open(sys.argv[1],'r').readlines()[0]
save_ = open(sys.argv[1]+'_re','w')
#print(ori_[:200])
a = '[' + ori_[:-1]+']'
#a =  ori_[:-1]
save_.write(a)

ff = open(sys.argv[1]+'_re')
fff = json.loads(a)
print('11')
with open(sys.argv[1]+'.json',"w") as f:
    json.dump(fff,f)
