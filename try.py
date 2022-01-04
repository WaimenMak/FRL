# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 21:00
# @Author  : Weiming Mai
# @FileName: try.py
# @Software: PyCharm

from multiprocessing import Process, Pipe
from threading import Thread
import os, time
# import psutil


class agent():
    def __init__(self):
        self.name = None
        self.a = None
        self.b = None

def client(agent, pipe2):
    # for i in range(2):
    #     print('我是子进程，我正在运行中')
    #     time.sleep(3)
    a, b = pipe2.recv()
    print(f"client{agent.name} recv {a} and {b}")

    for i in range(6):
        # print(f"hi!")

        a += 1
        if (i+1) % 2 == 0:
            pipe2.send((a, 0))
            print(f"client{agent.name} send a:{a}")

            a = pipe2.recv()
            # print(f"client{agent.name} recv {a}")
            agent.a = a
        if (i+1) % 3 == 0:
            b += 1
            pipe2.send((b, 1))
            print(f"client{agent.name} send b:{b}")
            b = pipe2.recv()


def server(pipe_dict, client_num, serving):
      #0:send, 1:recv
    a = 1
    b = 1
    target = 0
    local_model = []
    for i in range(client_num):
      pipe_dict[i][1].send((a,b))   #init model

    for j in range(5):


        for i in range(client_num):
            model, target = pipe_dict[i][1].recv()
            local_model.append(model)  #collect model
            print(f"model:{local_model}")
        if target == 0:   # two option
            a = sum(local_model)

            serving.a = a

            for i in range(client_num):
                # pipe_dict[i][1].send(a)
                pipe_dict[i][1].send(a)         # send model

                print(f"server send a:{a}")

        else:
            b = sum(local_model)
            serving.b = b
            for i in range(client_num):
                # pipe_dict[i][1].send(a)
                pipe_dict[i][1].send(b)  # send model

                print(f"server send b:{b}")
        local_model.clear()
class ser:
    def __init__(self):
        self.a = 0

def aa(pipe):
    # print(ser.a)
    for i in range(500):
        pipe.send([1,5646,767,0.555,6777])
    # time.sleep(2)
    #     pipe.send(2)
    # print(ser.a)

def ss(pipe, ser):
    a = 0
    time.sleep(3)
    for i in range(1000):
        if pipe.poll():
            a = pipe.recv()
            print(a)
            ser.a += a[0]

from models.Network import MLP
from utils.Tools import try_gpu
def gg(pipe,mlp):
    v = mlp.to(try_gpu())
    n = v.to("cpu")
    pipe.send(n)

if __name__ == '__main__':
    # 创建并启动子进程
    pipe1, pipe2 = Pipe()
    process_num = 2
    # serving = agent()
    # serving.name = "server"
    # agent1 = agent()
    # agent1.name = 1
    # agent2 = agent()
    # agent2.name = 2
    # pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (Pipe(),))
    ser = ser()
    # p = Thread(target=server, args=(pipe_dict, process_num, serving))
    p = Thread(target=ss, args=(pipe1, ser))
    # p2 = Process(target=client, args=(ss,pipe_dict[0][0]))
    p3 = Process(target=aa, args=(pipe2,))
    # mlp = MLP(2,1)
    # pipe1, pipe2 = Pipe()
    # p = Thread(target=gg, args=(pipe2, mlp))
    # p.start()
    # b = pipe1.recv()
    # print(b.state_dict())
    p.start()
    p3.start()
    [oo.join() for oo in [p,p3]]
    print(ser.a)
    # print(ser.b)
    # pid = p.pid  # 获取子进程的pid

    # 测试暂停子进程
    # time.sleep(1)
    # pause = psutil.Process(pid)  # 传入子进程的pid
    # pause.suspend()  # 暂停子进程
    # print('子进程暂停运行')
    # time.sleep(9)
    # pause.resume()  # 恢复子进程
    print('\n子进程已恢复运行')
