import sys

ori_ = open(sys.argv[1],'r').readlines()

save_ = open(sys.argv[1]+'_parse','w')

cur_img = ' '
for line in ori_:
    if not line.strip(): continue
    if 'GROUND' in line:
        cur_img = line.strip().split(':')[-1].strip()
        save_.write('img:'+cur_img+'\n'+'GROUND:'+'\n')
    elif 'label' in line and 'score' not in line:
        s_p_g = line.strip().split()
        save_.write('gted:'+' '+' '.join([s_p_g[1],s_p_g[3],s_p_g[5],s_p_g[7],s_p_g[9]])+'\n')
    elif 'PREDICTIONS' in line:
        save_.write('PRE:'+'\n')
    else:
        s_p = line.strip().split()
        save_.write('pred:'+' '+' '.join([s_p[2],s_p[4],s_p[5],s_p[7],s_p[9],s_p[11]])+'\n\n')
