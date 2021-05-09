# https://dengjunquan.github.io/posts/2018/08/DoAEstimation_Python/
# https://github.com/dengjunquan/DoA-Estimation-MUSIC-ESPRIT

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import scipy.signal as ss


class MUSIC_spectrum_with_Eve:

    def __init__(self, n_Rx, n_Tx,
                 num_angles, list_of_SNRs, list_of_DOAs,
                 kappa, n_NLOS_paths, max_delta_theta):
        # np.random.seed(6)
        self.PI = np.pi
        self.SNRs = list_of_SNRs  # signal-to-noise ratio
        self.n_Rx = n_Rx  # number of ULA elements = NA
        self.n_Tx = n_Tx   # number of transmitters
        # DOAs = np.random.uniform(-PI/2, PI/2, n_Tx)  # the nominal DOAs of all transmitters
        self.DOAs_in_degree = np.array(list_of_DOAs)  # convert list to array
        self.DOAs = self.DOAs_in_degree*self.PI/180

        # Path loss
        self.PL_LOS = kappa/(kappa + 1)  # k/(k+1) goes to 1 when k is large enough
        self.PL_NLOS = 1/(kappa + 1)  # for example, k=19, then 1/(k+1) = 1/20

        # NLOS components
        self.num_NLOS_paths = n_NLOS_paths
        self.max_delta_theta = max_delta_theta*self.PI/180  # in radian

        # For calculating the sample covariance matrix
        self.num_samples = 2**6  # this is also # of time slots per window

        # For plotting all spectrums versus all angles
        self.num_angles = num_angles
        self.angles = np.linspace(-self.PI/2, self.PI/2, self.num_angles)

        # n_Tx transmitted signals are concatenated in a column vector
        self.signals = np.sqrt(0.5)*(np.random.randn(self.n_Tx, 1)
                                     + 1j*np.random.randn(self.n_Tx, 1))  # signals from Bob and Eve

        # sample covariance matrix
        self.CovMat = self.sample_covariance()

    """Functions"""

    def response_to_an_angle(self, theta):
        # array holds the positions of antenna elements
        array = np.linspace(0, (self.n_Rx-1)/2, self.n_Rx)
        array = array.reshape([self.n_Rx, 1])
        v = np.exp(-1j*2*self.PI*array*np.sin(theta))
        return v/np.sqrt(self.n_Rx)

    """MUSIC and ESPRIT"""

    def music(self):
        eig_values, eig_vectors = LA.eig(self.CovMat)
        Qn = eig_vectors[:, self.n_Tx:self.n_Rx]  # a matrix associated with noise
        pspectrum = np.zeros(self.num_angles)
        for i in range(self.num_angles):
            theta = self.angles[i]
            av = self.response_to_an_angle(theta)
            # pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
            # pspectrum[i] = 1/LA.norm((np.conj(Qn.T) @ av))
            pspectrum[i] = 1/LA.norm(np.conj(av.T) @ Qn)
        psindB = np.log10(10*pspectrum / pspectrum.min())
        DoAsMUSIC, _ = ss.find_peaks(psindB, height=1.25, distance=1.5)
        return DoAsMUSIC, pspectrum  #DOAsMUSIC is of integer type
        # in radian and in dB, respectively

    def esprit(self):
        eig_values, eig_vectors = LA.eig(self.CovMat)
        S = eig_vectors[:, 0:self.n_Tx]
        Phi = LA.pinv(S[0:self.n_Rx-1]) @ S[1:self.n_Rx]
                # the original array is divided into two subarrays
                # [0,1,...,n_Rx-2] and [1,2,...,n_Rx-1]
        eigs, _ = LA.eig(Phi)
        DoAsESPRIT = np.arcsin(np.angle(eigs)/self.PI)
        return DoAsESPRIT  # in radian

    """Sample Covariance Matrix"""

    def sample_covariance(self):
        H = np.zeros([self.n_Rx, self.num_samples], dtype='complex128')
        for iter in range(self.num_samples):
            htmp = np.zeros([self.n_Rx, 1])
            for i in range(self.n_Tx):
                pha_i = np.exp(1j*2*self.PI*np.random.rand(1))
                signal_i = self.signals[i]
                DOA_i = self.DOAs[i]
                SNR_i = self.SNRs[i]
                LOS_component = signal_i * self.response_to_an_angle(DOA_i) * pha_i
                NLOS_components = 0
                for j in range(self.num_NLOS_paths):
                    DOA_NLOS_j = np.random.uniform(-self.max_delta_theta,
                                                   self.max_delta_theta)
                    NLOS_components += signal_i * self.response_to_an_angle(DOA_NLOS_j) * pha_i
                # end of for_j loop
                htmp = htmp + np.sqrt(SNR_i)*(
                                                    np.sqrt(self.PL_LOS)*LOS_component
                                                    + np.sqrt(self.PL_NLOS)*NLOS_components
                                                )
            # end of for_i loop
            noise = np.sqrt(0.5)*(np.random.randn(self.n_Rx, 1)
                                  + 1j*np.random.randn(self.n_Rx, 1))
            received_signal = htmp + noise
            H[:, iter] = received_signal.reshape(self.n_Rx)
        CovMat = H @ np.conj(H.T)  # np.matmul(H, np.conj(H).T)
        CovMat = (1/(self.num_samples-1)) * CovMat
        return CovMat

    """Correlation"""
    def correlation(self):
        # channel response
        h = np.zeros([self.n_Rx, 1])
        for i in range(self.n_Tx):
            signal_i = self.signals[i]
            DOA_i = self.DOAs[i]
            vi = self.response_to_an_angle(DOA_i)  # response is a vector
            h = h + signal_i * vi
        # correlation
        hv = np.zeros([self.num_angles, 1])
        for i in range(self.num_angles):
            angle_i = self.angles[i]
            vi = self.response_to_an_angle(angle_i)
            hv[i] = np.abs(np.conj(vi.T) @ h)  # np.abs(np.inner(h, av.conj()))
        #
        powers = np.zeros([self.n_Tx, 1])
        for i in range(self.n_Tx):
            DOA_i = self.DOAs[i]
            vi = self.response_to_an_angle(DOA_i)
            powers[i] = np.abs(np.conj(vi.T) @ h)  # np.abs(np.inner(h, av.conj()))
        return hv, powers

    """Plotting figures"""

    def plot_fig(self):
        angles_in_degree = self.angles*180/self.PI
        hv, powers = self.correlation()
        DoAs_MUSIC, pspectrum_in_dB = self.music()
        DoAsESPRIT = self.esprit()#*180/3.14
        
        #
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(angles_in_degree, hv, color='g')
        plt.plot(self.DOAs_in_degree, powers, '*')
        plt.title('Correlation')
        plt.legend(['Correlation power', 'Actual DoAs'])
        plt.xlabel('Angle (in degree\u00b0)', fontsize=12)
        plt.ylabel('Spectrum', fontsize=12)
        #
        plt.subplot(212)
        plt.plot(angles_in_degree, pspectrum_in_dB)
        plt.plot(angles_in_degree[DoAs_MUSIC], pspectrum_in_dB[DoAs_MUSIC],
                 'x', color='r')
        plt.title('MUSIC')
        plt.legend(['pseudo spectrum', 'Estimated DoAs'])
        plt.xlabel('Angle (in degree\u00b0)', fontsize=12)
        plt.ylabel('Spectrum', fontsize=12)
        fig.tight_layout()
        return None

# ============================================================================