
Everything is runned with DAQ_extractor.py:

Raw FP data is moved to the session folder. These are the raw data files that matter:
  20210315_1411.tdms : DAQ data can be loaded with package nptdms (https://pypi.org/project/npTDMS/)
  FP415 : FP data from every fiber in bundle for isosbestic band (type csv)
  FP470 : FP data from every fiber in bundle for gcamp band (type csv)

Pseudocode explanation:
Load FP data (line 51)
Load DAQ bpod and FP channel (line 72):
  FP frames send TTLs to analog 0.
  Bpod sends TTLs to analog 1.
Patch session if needed (accidental start of FP system twice etc) (line 83)
Assert that the number of TTL from FP is the same as the number of FP frames saved (FP470.csv file)* (line 104)
  Some tolerance is allowed (6 TTLS). FP system seems to send a few TTL before it starts saving them (This has been seen both at CCU and Princeton). We believe these are at the   end. (line 110 and 153)
Add DAQ times for every  FP frame. (line 106)
Extract bpod events from DAQ channel. Width of pulses gives you task logic (line 112).
Synchronize DAQ time with bpod time by aligning feedback times (line 126):
  Interpolation of bpod times for DAQ
  Add a bpod timestamp to every FP fluorescence measurement

