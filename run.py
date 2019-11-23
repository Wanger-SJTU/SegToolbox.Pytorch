import os
import logging

from threading import Thread
from collections import deque

root = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO,
                    filename='./voc_resnet18_dilation_fcn.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

task_que = deque()
running_done = deque(maxlen=2)

ratios   = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
# pretrain = [True, False]
# configs  = [os.path.join('configs', item) for item in os.listdir('configs') if '.' in item]

configs  = [
    #'configs/voc_resnet18_fcn.yaml',
     'configs/voc_resnet18_dilation_fcn.yaml'
]

meta_task = "python get_lr_value.py "

meta_task = "python main.py "

def call_fuc(command):
    global running_done
    running_done.append(command)
    logging.info("{} done.".format(command))

def running_task(command:str, gpu=0, call_back=call_fuc):
    global meta_task
    command +=  " --gpu_ids {0}".format(gpu)
    logging.info("running {} ".format(command))
    # subprocess.call(args=meta_task, executable=True)#+[command])
    try:
        ret=os.system(meta_task+ command)
    except Exception:
        logging.error("err in {}".format(command))
    if ret != 0 :
        logging.error("err in {}".format(command))

    call_fuc(command)

def generate_task():
    global task_que
    for ratio in ratios:
        # for pre in pretrain:
        for config in configs:
            para = " --ratio {0} --pretrain {1} --config {2}".format(
                ratio, "True", config)
            task_que.append(para)

def eval_task():
    global task_que
    for pre in pretrain:
        for config in configs:
            para = "--hist False --config {0} --pretrain {1}".format(config, pre)
            task_que.append(para)

if __name__ == "__main__":
    logging.info("start training")
    generate_task()
    for item in task_que:
        logging.info(item)
    # eval_task()
    logging.info("get {} tasks".format(len(task_que)))
    assert len(task_que) > 0
    t = Thread(target=running_task, kwargs={"command":task_que.pop(),"gpu":0})
    t.start()
    # t = Thread(target=running_task, kwargs={"command":task_que.pop(),"gpu":1})
    # t.start()

    while task_que:
        while running_done and task_que:
            last_command = running_done.pop().split(' ')
            t = Thread(target=running_task, kwargs={"command":task_que.pop(),"gpu":last_command[-1]})
            t.start() 