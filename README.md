![toolkit5G-logo](./logo/toolkit5G_Git.svg) 

# toolkit5G

5G Tookit provides a rich set of 3GPP standards compliant modules and libraries. 
These modules can be used for reseach and development on physical channels and procedures in uplink and downlink.
For more details follow our 5G Toolkit [webpage](https://gigayasawireless.github.io/toolkit5G/)

# Features Highlight: 5G Toolkit R24a

* **Uplink** and **Downlink** Chains
  * Provides **high-level** and **low level** **modules** for all the uplink downlink chains.
  * Support shared channels, control channels, broadcast channels and random access channels.
  * Modular design to support **plug** and **play**.
* **Artificial Intelligence/Machine learning** (AI/ML) for air interface
  * AI/ML for **Positioning**.
  * AI/ML for **Beam-management**.
  * AI/ML for **CSI feedback** (compression and reconstruction)
* 5G Physical Layer Procedures 
  * **Network Synchronization**: DL synchronization using SSB and UL synchronization using PRACH. 
  * **HARQ**: Based on both chase combining and incremental redundancy via rate matching. 
  * **Positioning**: Estimate the position of a device using 5G reference signals in UL/DL.
  * **Beam management**: P1-Procedure based on SSB, P2 procedure based on CSI-RS. 
  * CSI Reporting based CSI-RS:
    * **Link Adaptation**: based on estimated and reported CQI. 
    * **Rank Adaptation**: based on estimated and reported RI. 
    * **Precoding**: based on estimated and reported Type-I/II Codebook PMI. 
    * **Beam refinement**: based on reported CSI-RSRP/RSRQ.
    * **Mobility** management.
    * **Radio Resource** Management.
* Reference Signals:
  * All the reference sequences used in 5G till release are supported. 
  * **Downlink Synchronization**: PSS, SSS, DMRS for PBCH. 
  * **Uplink Synchronization**: PRACH. 
  * Data Channels: **DMRS** for PDSCH, PUSCH and, PSSCH. 
  * Control Channels: **DMRS** for PDCCH, PUCCH and, PSCCH. 
  * **Channel sounding**: CSI-RS, SRS. 
  * **Remote interference management** and **cross link interference**: RIM-RS 
  * **Positioning**: PRS. 
  * **Sidelink** reference signals: S-PSS, S-SSS, DMRSs and S-SSB.
* Symbol Mapping and Demapping
  * Supports all the symbol mapping schemes defined by 3GPP.
  * \frac{\pi}{2} -BPSK, BPSK, all the QAMs.
* Forward Error Correction
  * **LDPC** codes, **Polar** codes, **Reed-Muller** Codes and **Hamming** Codes.
  * Compliant with **3GPP**-standards.
  * Repetition codes will be provided in upcoming versions.
  * All the rate matching implementations are supported.
* Other standards compliant Modules
  * **Scramblers** for all the chains.
  * **Interleavers** for all the chains.
  * **Layer Mappers** for Shared Chains.
* **OFDM** and **Resource mapping**
  * Standards complaint OFDM implementation.
  * Resource Mapping for all the Physical channels.
  * Resource Mapping for all the Reference Signals.
* 3GPP Channel Models
  * Multi-cell **Massive** **MIMO** Simulations
  * Supports 3D-MIMO with single and dual **polarized** **Antennas**
  * **Antenna** with Custom Radiation Patterns
  * 3D **Mobility** Support
  * **Multi-frequency** Simulations
  * **OFDM** Channel
  * **Spatial Consistency** and spatial **correlations**
  * Support for:
    * **HAPS/Drone** Channels
    * **D2D** and **V2X** channels
    * **Non-terrestrial** (NTN) Channels
    * Sub-band full Duplex (SBFD)
* Sidelink Channels
  * Physical Sidelink Shared Channel (**PSSCH**)
  * Physical Sidelink Control Channel (**PSCCH**)
  * Physical Sidelink Broadcast Channel (**PSBCH**)
* Positioning in 5G Networks:
  * **Time** measurements based Positioning 
  * **DL-TDoA**, **UL-TDoA**, **m-RTT**, **ToA** based Positioning 
  * **Angle** measurements based Positioning 
  * **DL-AoD**, **UL-AoA**. 
  * **Hybrid** Positioning 
  * **AI** based **End to End** or **Direct** Positioning. 
  * **AI** **assisted** Positioning
* Integration Support with **Software Defined Radios** (SDRs)
  * SDR which supports **Python** based **API**  can be easily integrated. 
  * Few of the SDRs which have been integrated and well tested with the toolkit are listed.
    * Analog Devices: **Pluto** and **Phaser**, 
    * NI/Ettus Research: **USRP**-**B-210/205 mini/200**.
    * Others: **Lime** SDR, **RF-NM**.
  * Integration with **Signal vector/waveform generator** and **signal waveform analyzers**.
* 5G Configurations
  * Ease of generating parameters based on 3GPP 5G-NR specifications.
  * **Time Frequency** Configuration
  * **PDSCH** Configurations
  * **PBCH/SSB** Configurations.
  * **CSI-RS/SRS** Configuration

