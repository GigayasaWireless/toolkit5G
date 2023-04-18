import numpy  as np
import sionna as sn
import tensorflow as tf
from sionna.channel.tr38901 import AntennaArray, UMa
from sionna.channel         import gen_single_sector_topology as gen_topology

class UMaChannel():
    def __init__(self, nBatch, numUEs, Nfft, Neff, scs, lengthCP):
        self.nBatch    = nBatch
        self.numUEs    = numUEs
        self.Nfft      = Nfft
        self.Neff      = Neff
        self.scs       = scs
        self.lengthCP  = lengthCP
        
        self.scenario  = "uma"
        self.direction = "downlink"
        self.seed      = np.random.randint(2**16)
    
    def __call__(self, carrierFrequency, Nt_y, Nt_x, Pt, Nr_y, Nr_x, Pr, enable_pathloss=True):
        tf.random.set_seed(self.seed)

        txGridObj_ch  = sn.ofdm.ResourceGrid(num_ofdm_symbols = 1, 
                                             fft_size = self.Nfft, 
                                             subcarrier_spacing = self.scs, 
                                             num_tx = 1, 
                                             num_streams_per_tx   = 1, 
                                             cyclic_prefix_length = self.lengthCP, 
                                             num_guard_carriers   = (int((self.Nfft-self.Neff)/2), int((self.Nfft-self.Neff)/2)), 
                                             dc_null = False, 
                                             pilot_pattern = None, 
                                             pilot_ofdm_symbol_indices = None,
                                             dtype = tf.complex64)

        if(Pt==1):
            # Define the BS antenna array
            bs_array = AntennaArray(num_rows=Nt_y, num_cols=Nt_x, polarization="single",
                                    polarization_type="V", antenna_pattern="38.901",
                                    carrier_frequency=carrierFrequency) 
        else:
            # Define the BS antenna array
            bs_array = AntennaArray(num_rows=Nt_y, num_cols=Nt_x, polarization="dual",
                                    polarization_type="cross", antenna_pattern="38.901",
                                    carrier_frequency=carrierFrequency)

        if(Pr==1):
            # Define the BS antenna array
            ut_array = AntennaArray(num_rows=Nr_y, num_cols=Nr_x, polarization="single",
                                    polarization_type="V", antenna_pattern="38.901",
                                    carrier_frequency=carrierFrequency)
        else:
            # Define the BS antenna array
            ut_array = AntennaArray(num_rows=Nr_y, num_cols=Nr_x, polarization="dual",
                                    polarization_type="cross", antenna_pattern="38.901",
                                    carrier_frequency=carrierFrequency)


        # Create channel model
        channel_model = UMa(carrier_frequency=carrierFrequency, o2i_model="low",
                            ut_array=ut_array, bs_array=bs_array, direction=self.direction,
                            enable_pathloss=enable_pathloss, enable_shadow_fading=True)

        # Generate the topology
        topology = gen_topology(self.nBatch, self.numUEs, self.scenario)

        # Set the topology
        channel_model.set_topology(*topology, )

        # Visualize the topology
#         channel_model.show_topology()

        # OFDM Channel Generation 
        H_obj  = sn.channel.GenerateOFDMChannel(channel_model, txGridObj_ch, 
                                                normalize_channel=False, dtype = tf.complex64)
        H_ofdm = H_obj(1).numpy()
        # H_ofdm = H_ofdm/np.average(np.abs(H_ofdm))

        ht = np.fft.ifft(np.fft.ifftshift(H_ofdm, axes=-1), self.Nfft, axis= -1, norm="ortho").astype(dtype=np.complex64)
        ht[...,int(0.5*self.Nfft):] = 0
        
        return ht