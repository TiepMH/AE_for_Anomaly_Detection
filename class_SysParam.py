class SystemParameters:
    def __init__(self):
        self.n_Rx = 10
        self.n_Tx = 2
        ###
        self.snrdB_Bob = 5
        self.snrdB_Eve = 3
        ###
        self.SNR_Bob = 10**(self.snrdB_Bob/10)
        self.SNR_Eve = 10**(self.snrdB_Eve/10)
        self.list_of_SNRs = [self.SNR_Bob, self.SNR_Eve]
        ###
        self.DOA_Bob = - 40.5  # from -90 degree to +90 degree
        self.DOA_Eve = 10  # from -90 degree to +90 degree
        self.list_of_DOAs = [self.DOA_Bob, self.DOA_Eve]
        self.num_angles = 180
        ###
        self.Rician_factor = 2
        self.n_NLOS_paths = 10
        self.max_delta_theta = 90.0  # 90 degree


# =============================================================================
""" Save the system parameters as an object for later use """
# import pickle
# SysParam = SystemParameters()
# # save the SysParam object as a pickle-type file
# with open('input/mySysParam.pickle', 'wb') as f:
#     pickle.dump(SysParam, f)

# # load the pickle-type file
# with open('input/mySysParam.pickle', 'rb') as f:
#     SysParam = pickle.load(f)
