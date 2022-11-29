import _thread
import os
import configargparse
pid_name=os.popen('ps aux | grep transfer | awk \'{print $2}\' ').read()
#print(type(pid_name))
c=[]
pid_list=[]
for i in range(len(pid_name)):
    
    if pid_name[i]=='\n':
        pid_list.append("".join(c))
        c=[]
    elif pid_name[i]==' ':
        continue

    else:
        c.append(pid_name[i])
#print(pid_list)
for i in range(len(pid_list)):
    if int(pid_list[i])<10:
        continue
    os.system('kill -9 '+pid_list[i])
    print('kill -9 '+pid_list[i])
