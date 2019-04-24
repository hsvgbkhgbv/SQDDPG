'''
Currently, model parameters and data(replay buffer, stats) not shared. Should be shared in the future. 
'''
import time
from utilities.util import *
import torch
import torch.multiprocessing as mp

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, save_path, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm
        self.save_path = save_path

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, t = task

            if task == 'quit':
                return

            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch_mp()

            elif task == 'compute_grad':
                self.trainer.action_optimizer.zero_grad()
                self.trainer.value_optimizer.zero_grad()
                self.trainer.compute_grad(t)


            elif task == 'send_grad':
                action_grads = []
                for p in self.trainer.behaviour_net.action_dict.parameters():
                    if p._grad is not None:
                        action_grads.append(p._grad.data)
                value_grads = []
                for p in self.trainer.behaviour_net.value_dict.parameters():
                    if p._grad is not None:
                        value_grads.append(p._grad.data)
                self.comm.send([action_grads,value_grads])

            elif task == 'update_parameters':
                self.trainer.update_parameters(self.save_path)

class MultiProcessTrainer(object):
    def __init__(self, args, argv, trainer_maker):
        self.comms = []
        # create global trainer
        self.trainer = trainer_maker()
        # share network for all processes
        # self.trainer.share_memory()
        # itself will do the same job as workers
        self.nprocesses = argv.nprocesses
        self.save_path = argv.save_path
        self.nworkers = self.nprocesses - 1
        
        # create local workers
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, self.save_path, seed=2019)
            worker.start()

        self.action_grads = None
        self.value_grads = None
        self.worker_grads = None

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        if self.action_grads is None:
            self.action_grads = []
            for p in self.trainer.behaviour_net.action_dict.parameters():
                if p._grad is not None:
                    self.action_grads.append(p._grad.data)

        if self.value_grads is None:
            self.value_grads = []
            for p in self.trainer.behaviour_net.value_dict.parameters():
                if p._grad is not None:
                    self.value_grads.append(p._grad.data)

        # only need perform this once
        if self.worker_grads is None:
            self.worker_grads = []
            for i, comm in enumerate(self.comms):
                comm.send("send_grad")    
                self.worker_grads.append(comm.recv())
            print("Received WORKER {} gradients.".format(i))
            
        # accumulate grads from workers
        for g in self.worker_grads:
            ag = g[0]
            vg = g[1]
            for i in range(len(self.action_grads)):
                self.action_grads[i] += ag[i]
            for i in range(len(self.value_grads)):
                self.value_grads[i] += vg[i]


    def run_batch(self):

        # run workers in parallel
        for comm in self.comms:
            comm.send('run_batch')

        # run its own trainer
        batch, stat = self.trainer.run_batch_mp()

        return batch, stat 

    def train_batch(self, t, batch, stat):

        # compute grads in parallel
        for i, comm in enumerate(self.comms):
            comm.send(['compute_grad',t])

        # compute its own gradeint
        self.trainer.action_optimizer.zero_grad()
        self.trainer.value_optimizer.zero_grad()
        self.trainer.compute_grad(t)

        # obtain workers' grads pointers
        self.obtain_grad_pointers()

        # update parameters
        self.trainer.action_optimizer.step()
        self.trainer.value_optimizer.step()

        # broadcast global parameters
        PATH = self.save_path + 'model_save/tmp_model.pt'
        self.save_model(PATH)
        # write parameters in tmp_file
        for i, comm in enumerate(self.comms):
            comm.send('update_parameters')

        return self.trainer.stats

    def save_model(self,PATH):
        torch.save(self.trainer.behaviour_net.state_dict(),PATH) 

