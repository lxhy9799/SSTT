from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
import time
import hdf5storage
import mat73

class ngsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  #
        self.t_f = t_f  #
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.alltime = 0
        self.count = 0

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # agent id
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 11:]  #  grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1
        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t) #得到该帧在Track的索引
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut
    #
    # def getMean(self,vehId, t, dsId):
    #     vehTrack = self.T[dsId - 1][vehId - 1].transpose()
    #     refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
    #     stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
    #     enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
    #     fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
    #     mean_fut=
    #     return fut
    ## Collate function for dataloader

    def get_different_traffic(self,grid):
        nbrs_num=len(np.unique(grid))-1
        different_traffic=['light','moderate','heavy']
        if nbrs_num<=5:
            return different_traffic[0]
        elif 5<nbrs_num<=10:
            return different_traffic[1]
        elif 10<nbrs_num:
            return different_traffic[2]

    def collate_fn(self, samples):
        ttt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        #  mask_batch
        # self.alltime += (time.time() - ttt)
        # self.count += args['num_worker']
        #if (self.count > args['time']):
        #    print(self.alltime / self.count, "data load time")
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position


class highdDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        # self.D = np.transpose(h5py.File(mat_file, 'r')['traj'].value)
        # self.T = h5py.File(mat_file, 'r')['tracks']
        # self.T=mat73.loadmat(mat_file)['traj']
        self.t_h = t_h  #
        self.t_f = t_f  #
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # agent id
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 11:]  #  grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        nowt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # (len,batch*车数，2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 3)  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position



# Dataset class for the rounD dataset
class roundDataset(Dataset):

    def __init__(self, mat_file, t_h=50, t_f=100, d_s=4,
                 enc_size=64,  lat_dim=8,
                 lon_dim=3):

        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.A = scp.loadmat(mat_file)['anchor_traj_raw']

        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

        self.enc_size = enc_size  # size of encoder LSTM
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.ip_dim=3
        # self.goal_dim = goal_dim
        # self.en_ex_dim = en_ex_dim
    def __len__(self):
        return len(self.D)


    def __getitem__(self, idx):
        # print('getitem is called ')
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,15:] #14 if no entry_exit_class 15 if there
        neighbors = []

        # Encoding of Lateral and Longitudinal Intention Classes
        lat_class = self.D[idx, 12] - 1
        lat_enc = np.zeros([self.lat_dim])
        lat_enc[int(lat_class)] = 1

        lon_class = self.D[idx, 13] - 1
        lon_enc = np.zeros([self.lon_dim])
        lon_enc[int(lon_class)] = 1

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)
        fut, fut_anchored = self.getFuture(vehId, t, dsId, lat_class, lon_class)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            neighbors.append(self.getHistory(i.astype(int), t, vehId, dsId))

        return hist, fut, neighbors, lat_enc, lon_enc, dsId, vehId, t, fut_anchored

    # Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, self.ip_dim])
        else:
            veh_tracks = self.T

            if veh_tracks.shape[1] <= vehId - 1:
                return np.empty([0, self.ip_dim])
            refTrack = veh_tracks[dsId - 1][refVehId - 1].transpose()
            vehTrack = veh_tracks[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:self.ip_dim + 1]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, self.ip_dim])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, self.ip_dim])
            return hist

    # Helper function to get track future
    def getFuture(self, vehId, t, dsId, lat_class, lon_class):

        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:self.ip_dim + 1]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:self.ip_dim + 1] - refPos

        anchor_traj = self.A[int(lon_class), int(lat_class)]
        anchor_traj = anchor_traj[0:-1:self.d_s, :]

        fut_anchored = anchor_traj[0:len(fut), :] - fut

        return fut, fut_anchored

    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        # nbr_batch_size = 0
        nbr_batch_size = 0
        nbr_list_len = torch.zeros(len(samples),1)
        for sample_id , (_, _, nbrs, _, _, _, _, _, _) in enumerate(samples):
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
            nbr_list_len[sample_id] = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])

        # nbr_batch_size = int((sum(nbr_list_len)).item())
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, self.ip_dim)

        # Initialize social mask batch:

        mask_batch = torch.zeros(len(samples), 11, self.enc_size)  # (batch,9,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), self.ip_dim)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        ds_ids_batch = torch.zeros(len(samples), 1)
        vehicle_ids_batch = torch.zeros(len(samples), 1)
        frame_ids_batch = torch.zeros(len(samples), 1)
        lat_enc_batch = torch.zeros(len(samples), self.lat_dim)
        lon_enc_batch = torch.zeros(len(samples), self.lon_dim)
        fut_anchored_batch = torch.zeros(self.t_f // self.d_s, len(samples), self.ip_dim)
        count = 0
        for sampleId, (hist, fut, nbrs,lat_enc, lon_enc, ds_ids, vehicle_ids, frame_ids, fut_anchored) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            for k in range(self.ip_dim):
                hist_batch[0:len(hist), sampleId, k] = torch.from_numpy(hist[:, k])
                fut_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut[:, k])
                fut_anchored_batch[0:len(fut), sampleId, k] = torch.from_numpy(fut_anchored[:, k])

            op_mask_batch[0:len(fut), sampleId, :] = 1
            ds_ids_batch[sampleId, :] = torch.tensor(ds_ids.astype(np.float64))
            vehicle_ids_batch[sampleId, :] = torch.tensor(vehicle_ids.astype(np.float64))
            frame_ids_batch[sampleId, :] = torch.tensor(frame_ids.astype(int).astype(np.float64))
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    for k in range(self.ip_dim):
                        nbrs_batch[0:len(nbr), count, k] = torch.from_numpy(nbr[:, k])
                    pos = id % 11
                    mask_batch[sampleId, pos, :] = torch.ones(self.enc_size).byte()
                    count += 1

        return hist_batch, nbrs_batch, nbr_list_len , mask_batch, fut_batch, lat_enc_batch, \
               lon_enc_batch, op_mask_batch, ds_ids_batch, vehicle_ids_batch, frame_ids_batch, fut_anchored_batch